"""GRPO for the SQL agent, in MLX, with LoRA.

One step:
  1. sample `--prompts` tasks, roll each out `--group` times with the current policy
  2. advantage = reward - mean reward of its group (no std division: Dr. GRPO)
  3. drop groups where every rollout got the same reward (zero signal)
  4. one gradient step on  -sum(adv * logp(generated tokens)) / n_generated_tokens

Because we take a single gradient step per batch of fresh rollouts, the policy
that sampled is the policy being updated, so the PPO ratio is exactly 1 and the
clipped objective reduces to this plain policy gradient. No KL penalty. Teacher rescues add a supervised positive-gradient term;
those trajectories are not on-policy student samples.

Logs go to runs/<name>/ for the dashboard:
  config.json, metrics.jsonl, evals.jsonl, status.json, samples/step_NNNNN.json
"""

import argparse
import json
import math
import shutil
import signal
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters

import env
import evolve
from curriculum import Curriculum
from rollout import rollout, set_memory_budget
from teacher import Teacher, as_episode

HERE = Path(__file__).parent


class RunLog:
    def __init__(self, root):
        self.root = root
        (root / "samples").mkdir(parents=True, exist_ok=True)

    def append(self, name, row):
        with open(self.root / name, "a") as f:
            f.write(json.dumps(row) + "\n")

    def write(self, name, obj):
        # Write then rename so the dashboard never reads a half-written file.
        tmp = self.root / (name + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(obj))
        tmp.replace(self.root / name)

    def status(self, **kw):
        self.write("status.json", {"updated": time.time(), **kw})


def live_writer(log, phase, step, group_size):
    """Overwrite runs/<name>/live.json with every rollout in flight, for the dashboard's Live tab."""
    t0 = time.time()

    def write(episodes, partial):
        log.write("live.json", {
            "phase": phase, "step": step, "updated": time.time(), "seconds": time.time() - t0,
            "episodes": [{
                "group": i // group_size, "template": e.task.template, "question": e.task.question,
                "turns": e.turns, "segments": e.segments, "partial": partial.get(id(e)),
                "done": e.done, "outcome": e.outcome or None, "reward": e.reward,
            } for i, e in enumerate(episodes)],
        })
    return write


def episode_record(e, group, adv):
    return {
        "group": group, "template": e.task.template, "tier": e.task.tier,
        "question": e.task.question, "gold_sql": e.task.gold_sql,
        "reward": e.reward, "advantage": adv, "outcome": e.outcome, "turns": e.turns,
        "gen_tokens": sum(e.mask), "prose_tokens": e.prose_tokens, "segments": e.segments, "final_sql": e.final_sql,
        "literal": e.final_sql is not None and env.is_literal(e.final_sql),
        "verifier_reason": e.verifier_reason, "db_path": str(e.task.db_path),
        "verifier_version": env.VERIFIER_VERSION, "protocol_version": env.PROTOCOL_VERSION,
        "answer_rows": e.answer_rows, "gold_rows": e.gold_rows, "source": e.source,
    }


def seq_logprob_loss(model, inputs, positions, targets, adv, temp):
    # Only generated tokens get gradient, and they are ~20% of a trajectory, so
    # project just those hidden states onto the 130k-entry vocabulary.
    # Full-sequence logits would be the biggest tensor in the step by far.
    hidden = model.model(inputs)
    hidden = hidden.reshape(-1, hidden.shape[-1])[positions]
    logits = model.lm_head(hidden).astype(mx.float32) / temp
    tok = mx.take_along_axis(logits, targets[:, None], axis=-1).squeeze(-1)
    return -((tok - mx.logsumexp(logits, axis=-1)) * adv).sum()


def micro_batches(batch, max_tokens):
    """Group sequences so each forward/backward pass holds at most max_tokens (padded)."""
    batch = sorted(batch, key=lambda b: len(b[0]))  # similar lengths together = less padding
    chunk = []
    for item in batch:
        if chunk and (len(chunk) + 1) * len(item[0]) > max_tokens:
            yield chunk
            chunk = []
        chunk.append(item)
    if chunk:
        yield chunk


def train_step(model, opt, batch, temp, micro_tokens, max_grad_norm):
    """batch: list of (tokens, mask, advantage). Returns (loss, grad_norm, n_tokens)."""
    n_tokens = sum(sum(m[1:]) for _, m, _ in batch)
    loss_and_grad = nn.value_and_grad(model, seq_logprob_loss)
    # Recompute each layer's activations in the backward pass instead of storing
    # them all: without this a 4k-token micro-batch peaks near 20GB. It patches
    # the layer class, so restore it afterwards or decoding pays the overhead too.
    layer_cls = type(model.layers[0])
    plain_call = layer_cls.__call__
    grad_checkpoint(model.layers[0])
    try:
        grads, total = _accumulate(model, loss_and_grad, batch, temp, micro_tokens, n_tokens)
    finally:
        layer_cls.__call__ = plain_call
    grads, grad_norm = optim.clip_grad_norm(grads, max_grad_norm)
    if not math.isfinite(total) or not math.isfinite(grad_norm.item()):
        raise FloatingPointError("Nonfinite loss or gradients; update cancelled")
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)
    mx.clear_cache()
    return total, grad_norm.item(), n_tokens


def _accumulate(model, loss_and_grad, batch, temp, micro_tokens, n_tokens):
    grads, total = None, 0.0
    for chunk in micro_batches(batch, micro_tokens):
        L = max(len(t) for t, _, _ in chunk) - 1
        inputs = mx.array([t[:-1] + [0] * (L - len(t) + 1) for t, _, _ in chunk])
        # Token i+1 is predicted at position i, so the mask shifts with the targets.
        positions, targets, adv = [], [], []
        for row, (t, m, a) in enumerate(chunk):
            for i in range(len(t) - 1):
                if m[i + 1]:
                    positions.append(row * L + i)
                    targets.append(t[i + 1])
                    adv.append(a)
        loss, g = loss_and_grad(model, inputs, mx.array(positions), mx.array(targets),
                                mx.array(adv, dtype=mx.float32), temp)
        g = tree_map(lambda x: x / n_tokens, g)
        grads = g if grads is None else tree_map(mx.add, grads, g)
        mx.eval(loss, grads)
        total += loss.item() / n_tokens
    return grads, total


def rescue_groups(jobs, eps, args, tok, deadline):
    """For each group where every rollout failed, swap the last rollout for the
    teacher's verified solution, so the group has a correct example to move toward.
    Returns {group index: teacher Episode}."""
    rescue = {}
    for g in range(args.prompts):
        group = eps[g * args.group:(g + 1) * args.group]
        if any(e.outcome == "correct" for e in group):
            continue
        try:
            solution = jobs[group[0].task.question].result(timeout=max(0.0, deadline - time.time()))
        except TimeoutError:
            print(f"  teacher: no answer in time for group {g}")
            continue
        except Exception as e:  # network or API error: train this step without the rescue
            print(f"  teacher: {type(e).__name__}: {str(e)[:120]}")
            continue
        if solution["correct"]:
            try:
                rescue[g] = as_episode(tok, group[-1], solution, args.max_tokens)
            except ValueError as exc:
                print(f"  teacher: skipped group {g}: {exc}")
    return rescue


def save_checkpoint(model, opt, path, args, step, curriculum):
    """Write adapters, optimizer state and trainer state to a temporary directory, then
    rename it, so an interrupted save never leaves a partial checkpoint."""
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    tmp = path.with_name(path.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(tmp / "optimizer.safetensors"), dict(tree_flatten(opt.state)))
    (tmp / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": args.lora_layers,
        "lora_parameters": {"rank": args.rank, "scale": args.scale, "dropout": 0.0},
    }, indent=2))
    (tmp / "trainer_state.json").write_text(json.dumps({
        "step": step, "time": time.time(), "run": args.name,
        "verifier_version": env.VERIFIER_VERSION, "protocol_version": env.PROTOCOL_VERSION,
        "curriculum": {"rate": curriculum.rate, "seen": dict(curriculum.seen)},
    }))
    tmp.rename(path)


def load_optimizer(opt, model, path):
    opt.init(model.trainable_parameters())
    opt.state = tree_unflatten(list(mx.load(str(path / "optimizer.safetensors")).items()))


def git_revision():
    run = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "."))}


class Stop(Exception):
    """SIGTERM or SIGINT. Raised at once unless an update or save is in progress."""


class Signals:
    def __init__(self):
        self.critical, self.pending = False, None
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.handle)

    def handle(self, signum, _frame):
        self.pending = signal.Signals(signum).name
        if not self.critical:
            raise Stop(self.pending)

    def __enter__(self):
        self.critical = True

    def __exit__(self, *exc):
        self.critical = False
        if self.pending and exc[0] is None:
            raise Stop(self.pending)


def run_eval(model, tok, args, log, step):
    model.eval()
    log.status(phase="eval", step=step)
    eps, stats = rollout(model, tok, env.eval_tasks(args.eval_per_template, args.eval_k), temp=args.temp,
                         max_turns=args.max_turns, max_tokens=args.max_tokens,
                         show_schema=not args.hide_schema, batch_size=args.gen_batch,
                         live=live_writer(log, "eval", step, args.eval_k))
    by_t, tiers, splits = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in eps:
        ok = float(e.outcome == "correct")
        by_t[e.task.template].append(ok)
        tiers[e.task.tier].append(ok)
        splits[e.task.split].append(ok)
    row = {
        "step": step, "time": time.time(),
        "pass1": sum(e.outcome == "correct" for e in eps) / len(eps),
        "verifier_version": env.VERIFIER_VERSION, "protocol_version": env.PROTOCOL_VERSION,
        "questions": len({e.task.question for e in eps}), "attempts": len(eps),
        "evaluation_split": "unseen databases; held-out templates, reserved parameters, or shared wording",
        "by_split": {k: sum(v) / len(v) for k, v in sorted(splits.items())},
        "split_attempts": {k: len(v) for k, v in sorted(splits.items())},
        "heldout_templates": list(env.HELDOUT_TEMPLATES),
        "by_template": {k: sum(v) / len(v) for k, v in by_t.items()},
        "by_tier": {str(k): sum(v) / len(v) for k, v in sorted(tiers.items())},
        "outcomes": dict(Counter(e.outcome for e in eps)),
        "seconds": stats["seconds"],
    }
    log.write(f"eval_samples/step_{step:05d}.json",
              [episode_record(e, i // args.eval_k, 0.0) for i, e in enumerate(eps)])
    log.append("evals.jsonl", row)
    print(f"[eval step {step}] pass@1 {row['pass1']:.3f}  splits {row['by_split']}  tiers {row['by_tier']}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--model", default=str(HERE.parent / "models/MiniCPM5-1B"))
    ap.add_argument("--adapter", help="Warm-start adapter weights; starts a fresh optimizer")
    ap.add_argument("--resume", help="Checkpoint to continue: restores adapters, optimizer state, step and curriculum")
    ap.add_argument("--max-hours", type=float, default=0, help="stop before a step that would end after this wall-clock budget (0 = no limit)")
    ap.add_argument("--max-nonfinite", type=int, default=3, help="stop after this many consecutive skipped nonfinite updates")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts", type=int, default=16, help="distinct tasks per step")
    ap.add_argument("--group", type=int, default=8, help="rollouts per task")
    ap.add_argument("--tiers", default="1,2,3")
    ap.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True,
                    help="sample templates toward a ~50%% solve rate instead of uniformly")
    ap.add_argument("--evolve-every", type=int, default=10,
                    help="every N steps, generate new templates for the weakest skills (0 disables)")
    ap.add_argument("--evolve-skills", type=int, default=2, help="weakest skills to target each time")
    ap.add_argument("--evolve-n", type=int, default=2, help="templates proposed per skill")
    ap.add_argument("--evolve-model", default="glm-5.3-flash")
    ap.add_argument("--evolve-min-seen", type=int, default=16, help="rollouts of a skill needed before judging it weak")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--rank", type=int, default=32)
    # LoRA output multiplier (alpha / rank). mlx-lm's SFT default is 20, which made
    # RL updates ~10x too large here: held-out pass@1 halved within 2 steps.
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--lora-layers", type=int, default=24)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256, help="per assistant turn")
    ap.add_argument("--hide-schema", action="store_true")
    ap.add_argument("--micro-tokens", type=int, default=8192, help="padded tokens per forward/backward pass")
    ap.add_argument("--gen-batch", type=int, default=64, help="sequences decoded in parallel")
    ap.add_argument("--mem-gb", type=float, default=26, help="cap on MLX memory in GiB (this Mac has 36GB)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=25, help="0 disables")
    ap.add_argument("--eval-per-template", type=int, default=2)
    ap.add_argument("--eval-k", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-answer-reward", type=float, default=-0.2,
                    help="reward for terminating without submission, including token and turn limits")
    ap.add_argument("--prose-penalty", type=float, default=0.003,
                    help="reward lost per token of non-SQL text in correct answers (max 0.3); trains out apologies and narration")
    ap.add_argument("--teacher", default=None, help="OpenCode Go model that rescues all-fail groups; glm-5.3-flash solved 9/9 hard questions in 3s")
    ap.add_argument("--teacher-effort", default="low", help="teacher reasoning_effort: low is fastest that GLM-5.3 accepts")
    ap.add_argument("--teacher-wait", type=float, default=60, help="seconds to wait for teacher answers after rollouts")
    args = ap.parse_args()

    if args.adapter and args.resume:
        raise ValueError("Use --adapter or --resume, not both")
    start_time = time.time()
    resume_state = json.loads((Path(args.resume) / "trainer_state.json").read_text()) if args.resume else None
    if args.adapter or args.resume:
        adapter_config = json.loads((Path(args.adapter or args.resume) / "adapter_config.json").read_text())
        args.rank = adapter_config["lora_parameters"]["rank"]
        args.scale = adapter_config["lora_parameters"]["scale"]
        args.lora_layers = adapter_config["num_layers"]
    run = HERE / "runs" / args.name
    if run.exists():
        raise FileExistsError(f"Run already exists: {run}. Choose a new --name")
    log = RunLog(run)
    log.write("config.json", {**vars(args), "started": time.time(),
                             "verifier_version": env.VERIFIER_VERSION, "protocol_version": env.PROTOCOL_VERSION,
                             "optimizer_restored": bool(args.resume), "start_step": resume_state["step"] if resume_state else 0,
                             "evaluation_db": str(env.EVAL_DB_PATH), "git": git_revision()})
    log.status(phase="loading", step=0)
    set_memory_budget(args.mem_gb)

    env.ensure_databases()
    mx.random.seed(args.seed)
    model, tok = load(args.model)
    model.freeze()
    if args.adapter or args.resume:
        model = load_adapters(model, args.adapter or args.resume)
    else:
        linear_to_lora_layers(model, args.lora_layers, {"rank": args.rank, "scale": args.scale, "dropout": 0.0})
    model.eval()
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable params: {n_train / 1e6:.1f}M")
    opt = optim.Adam(learning_rate=args.lr)
    first_step = 1
    if resume_state:
        if resume_state["verifier_version"] != env.VERIFIER_VERSION:
            print(f"resuming a {resume_state['verifier_version']} checkpoint under {env.VERIFIER_VERSION}")
        load_optimizer(opt, model, Path(args.resume))
        first_step = resume_state["step"] + 1
    tiers = tuple(int(t) for t in args.tiers.split(","))
    teacher = Teacher(args.teacher, max_turns=args.max_turns, effort=args.teacher_effort,
                      show_schema=not args.hide_schema) if args.teacher else None
    generated = evolve.load_accepted()
    print(f"templates: {len(env.BUILTIN)} built in, {len(generated)} generated earlier")
    curriculum = Curriculum(hard_bonus=0.1 if teacher else 0.0)
    if resume_state:
        curriculum.rate = {k: v for k, v in resume_state["curriculum"]["rate"].items() if k in env.TEMPLATES}
        curriculum.seen.update({k: v for k, v in resume_state["curriculum"]["seen"].items() if k in env.TEMPLATES})
    # Template generation calls the API and validates with a solver; it runs beside
    # training and its results are picked up at the start of a later step.
    evolver_state = {"pool": ThreadPoolExecutor(1), "future": None, "spent": 0.0}

    signals = Signals()
    log.weights_step = first_step - 1
    step, last_saved, nonfinite_run, stop_reason = first_step - 1, None, 0, "completed"
    deadline = start_time + args.max_hours * 3600 if args.max_hours else None
    step_times, eval_s = [], 0.0
    try:
        if args.eval_every:
            eval_s = run_eval(model, tok, args, log, step)["seconds"]
        while step < args.steps:
            if deadline:
                # Leave room for the next step, its scheduled evaluation and the final one.
                expected = max(step_times[-3:], default=0) + eval_s * (1 + ((step + 1) % args.eval_every == 0 if args.eval_every else 0))
                if time.time() + expected > deadline:
                    stop_reason = "time limit"
                    break
            step += 1
            nonfinite_run = train_one_step(step, args, model, tok, opt, log, curriculum, teacher, evolver_state, tiers,
                                           signals, nonfinite_run)
            step_times.append(time.time() - log.step_started)
            if args.save_every and step % args.save_every == 0:
                with signals:
                    save_checkpoint(model, opt, run / f"step_{step:05d}", args, step, curriculum)
                last_saved = step
            if args.eval_every and step % args.eval_every == 0:
                eval_s = run_eval(model, tok, args, log, step)["seconds"]
        if args.eval_every and step % args.eval_every and step >= first_step:
            run_eval(model, tok, args, log, step)
        with signals:
            save_checkpoint(model, opt, run / "final", args, step, curriculum)
    except BaseException as exc:
        # Weights only change inside the protected update, so they are a valid state after weights_step.
        done = log.weights_step
        if done >= first_step and last_saved != done and not isinstance(exc, FileExistsError):
            signals.critical = True
            path = run / f"interrupted_step_{done:05d}"
            if not path.exists():
                save_checkpoint(model, opt, path, args, done, curriculum)
                print(f"saved {path}")
        raise
    finally:
        evolver_state["pool"].shutdown(wait=False, cancel_futures=True)
        if teacher:
            teacher.close()
    log.status(phase="done", step=step, stop_reason=stop_reason, elapsed_hours=(time.time() - start_time) / 3600)


def train_one_step(step, args, model, tok, opt, log, curriculum, teacher, evolver_state, tiers, signals, nonfinite_run):
    """One GRPO step. Returns the number of consecutive skipped nonfinite updates."""
    t0 = log.step_started = time.time()
    new_templates = []
    evolving = evolver_state["future"]
    if evolving and evolving.done():
        try:
            accepted, stats = evolving.result()
            for spec in accepted:
                env.add_template(spec)
            new_templates = [spec["name"] for spec in accepted]
            evolver_state["spent"] += stats["spent"]
            print(f"  evolve: {stats['accepted']} of {stats['proposed']} new templates accepted: {new_templates}")
        except Exception as e:  # API or network failure: keep training on the templates we have
            print(f"  evolve failed: {type(e).__name__}: {str(e)[:160]}")
        evolver_state["future"] = None
    tasks = env.sample_tasks(args.prompts, seed=args.seed * 1_000_003 + step, tiers=tiers,
                             weights=curriculum.weights() if args.curriculum else None)
    group_tasks = [t for t in tasks for _ in range(args.group)]

    def progress(turn, active, gen_tokens, secs):
        log.status(phase="rollout", step=step, turn=turn, active=active, gen_tokens=gen_tokens, seconds=secs)

    # The teacher works on the same questions over the network while the student
    # rolls out locally, so its latency hides behind compute we pay for anyway.
    teacher_jobs = teacher.submit(tasks) if teacher else {}
    log.status(phase="rollout", step=step, turn=0)
    eps, rstats = rollout(model, tok, group_tasks, temp=args.temp, max_turns=args.max_turns,
                          max_tokens=args.max_tokens, show_schema=not args.hide_schema,
                          batch_size=args.gen_batch, on_turn=progress,
                          live=live_writer(log, "rollout", step, args.group), prose_penalty=args.prose_penalty,
                          no_answer_reward=args.no_answer_reward)
    t_roll = time.time() - t0
    curriculum.update(eps)
    if args.evolve_every and step % args.evolve_every == 0 and not evolver_state["future"]:
        if weak := curriculum.weakest_skills(args.evolve_skills, min_seen=args.evolve_min_seen):
            evolver_state["future"] = evolver_state["pool"].submit(evolve.evolve, weak, args.evolve_n, args.evolve_model)
            print(f"  evolve: generating templates for {weak}")

    rescue = rescue_groups(teacher_jobs, eps, args, tok, deadline=time.time() + args.teacher_wait) if teacher else {}

    batch, records, kept = [], [], 0
    for g in range(args.prompts):
        group = eps[g * args.group:(g + 1) * args.group]
        if g in rescue:
            group = group[:-1] + [rescue[g]]
        mean = sum(e.reward for e in group) / len(group)
        student_mean = sum(e.reward for e in group if e.source == "student") / sum(e.source == "student" for e in group)
        if any(e.reward != mean for e in group):
            kept += 1
        for e in group:
            adv = e.reward - mean
            if g in rescue and e.source == "student":
                # Score students against each other, not against the teacher. Pushing
                # every failed attempt below the teacher taught the model to stop
                # querying ("answered without a query" went 48 -> 76 of 128 in three
                # steps); relative to each other, trying still beats not trying.
                adv = e.reward - student_mean
            records.append(episode_record(e, g, adv))
            if adv != 0:
                batch.append((e.tokens, e.mask, adv))

    log.status(phase="train", step=step)
    t1 = time.time()
    loss, grad_norm, n_tok, skipped = 0.0, 0.0, 0, False
    if batch:
        try:
            with signals:
                loss, grad_norm, n_tok = train_step(model, opt, batch, args.temp, args.micro_tokens, args.max_grad_norm)
            nonfinite_run = 0
        except FloatingPointError as exc:
            skipped, nonfinite_run = True, nonfinite_run + 1
            print(f"  {exc} ({nonfinite_run} in a row)")
            if nonfinite_run >= args.max_nonfinite:
                raise
    t_train = time.time() - t1
    log.weights_step = step

    rewards = [e.reward for e in eps]
    submits = [e for e in eps if e.final_sql is not None]
    literal = [e for e in submits if env.is_literal(e.final_sql)]
    by_t, by_tier = defaultdict(list), defaultdict(list)
    for e in eps:
        # Rates count correct submissions; reward shaping is reported by reward_mean.
        by_t[e.task.template].append(float(e.outcome == "correct"))
        by_tier[e.task.tier].append(float(e.outcome == "correct"))
    row = {
        "step": step, "time": time.time(),
        "reward_mean": sum(rewards) / len(rewards),
        "outcomes": dict(Counter(e.outcome for e in eps)),
        "by_template": {k: sum(v) / len(v) for k, v in by_t.items()},
        "by_tier": {str(k): sum(v) / len(v) for k, v in sorted(by_tier.items())},
        "turns_mean": sum(e.turns for e in eps) / len(eps),
        "literal_frac": len(literal) / max(len(submits), 1),
        "literal_correct_frac": sum(e.outcome == "correct" for e in literal) / max(sum(e.outcome == "correct" for e in eps), 1),
        "gen_tokens_mean": sum(sum(e.mask) for e in eps) / len(eps),
        "prose_tokens_mean": sum(e.prose_tokens for e in eps) / len(eps),
        "groups_kept": kept, "groups_total": args.prompts,
        "loss": loss, "grad_norm": grad_norm, "train_tokens": n_tok, "update_skipped_nonfinite": skipped,
        "rollout_s": t_roll, "train_s": t_train,
        "gen_tps": rstats["gen_tokens"] / max(rstats["seconds"], 1e-9),
        "peak_mem_gb": mx.get_peak_memory() / 1024**3,
    }
    row |= {"curriculum": curriculum.snapshot(), "new_templates": new_templates, "evolve_spent": evolver_state["spent"]}
    if teacher:
        dead = sum(1 for g in range(args.prompts)
                   if not any(e.outcome == "correct" for e in eps[g * args.group:(g + 1) * args.group]))
        row |= {"teacher_dead_groups": dead, "teacher_rescued": len(rescue), "teacher_spent": teacher.spent}
    mx.reset_peak_memory()
    log.append("metrics.jsonl", row)
    log.write(f"samples/step_{step:05d}.json", records)
    print(f"step {step:4d}  reward {row['reward_mean']:.3f}  kept {kept}/{args.prompts}  "
          f"turns {row['turns_mean']:.2f}  loss {loss:+.4f}  |g| {grad_norm:.3f}  "
          f"roll {t_roll:.0f}s train {t_train:.0f}s  {row['outcomes']}")
    return nonfinite_run


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Preserve the last phase/step but make failures visible to the dashboard.
        import sys
        if "--name" in sys.argv:
            name = sys.argv[sys.argv.index("--name") + 1]
            status_path = HERE / "runs" / name / "status.json"
            if status_path.exists() and not isinstance(exc, FileExistsError):
                status = json.loads(status_path.read_text())
                RunLog(status_path.parent).status(**{k: v for k, v in status.items() if k not in ("updated", "phase")},
                                                  phase="error", error=f"{type(exc).__name__}: {exc}")
        raise
