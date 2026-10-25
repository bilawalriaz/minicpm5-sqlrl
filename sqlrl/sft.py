"""Supervised distillation on verified solutions, before or between RL runs.

    python sft.py --name sft_v1 --resume runs/final_v23b/step_00200 \\
        --solutions data/sql-v2.2/solutions.jsonl --replay runs/final_v23b --epochs 2

Teaching examples: verified teacher solutions (scripts/collect_solutions.py) plus the
student's own correct transcripts from earlier runs, so new task types are learned
without losing old ones. Examples whose mean per-token NLL under the student exceeds
--max-nll are skipped: far-off solutions degraded existing skills during RL. The
selection and check sets are scored before and after; the checkpoint holds the
averaged weights (adapters.safetensors), raw weights and optimizer state for --resume.
"""
import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.utils import load_adapters

import env
import evolve
import train
from curriculum import Curriculum
from rollout import OBS_TEMPLATE, prompt_tokens, set_memory_budget

HERE = Path(__file__).parent


def episode_tokens(tok, question, turns, show_schema=True):
    """turns: [(assistant text, tool result)]; the last turn is the submission."""
    task = env.Task("sft", 0, question, "")
    tokens = list(prompt_tokens(tok, task, show_schema))
    mask = [0] * len(tokens)
    im_end = tok.encode("<|im_end|>", add_special_tokens=False)
    for i, (text, result) in enumerate(turns):
        ids = tok.encode(text, add_special_tokens=False) + im_end
        tokens += ids
        mask += [1] * len(ids)
        if i < len(turns) - 1:
            obs = tok.encode(OBS_TEMPLATE.format(result), add_special_tokens=False)
            tokens += obs
            mask += [0] * len(obs)
    return tokens, mask


def teacher_examples(tok, path, max_tokens):
    out = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["template"] in env.HELDOUT_TEMPLATES or r["question"] in env.eval_question_texts():
            continue
        turns = []
        for name, query, result in r["actions"]:
            value = f"<![CDATA[{query}]]>" if any(c in query for c in "<&\n") else query
            turns.append((f'<function name="{name}"><param name="query">{value}</param></function>', result))
        tokens, mask = episode_tokens(tok, r["question"], turns)
        if max(len(tok.encode(t, add_special_tokens=False)) for t, _ in turns) < max_tokens:
            out.append(("teacher", r["template"], tokens, mask))
    return out


def replay_examples(tok, runs, limit, seed):
    """Correct student transcripts from earlier runs, one per question, newest first."""
    seen, out = set(), []
    for run in runs:
        for path in sorted((HERE / run / "samples").glob("step_*.json"), reverse=True):
            for r in json.loads(path.read_text()):
                if r["source"] != "student" or r["outcome"] != "correct" or r["question"] in seen:
                    continue
                if r["template"] in env.HELDOUT_TEMPLATES or r["question"] in env.eval_question_texts():
                    continue
                segs = r["segments"]
                turns = [(segs[i]["text"], segs[i + 1]["text"] if i + 1 < len(segs) else "")
                         for i in range(0, len(segs), 2)]
                seen.add(r["question"])
                out.append(("student", r["template"], *episode_tokens(tok, r["question"], turns)))
    random.Random(seed).shuffle(out)
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--resume", help="checkpoint to start from (adapters, optimizer, curriculum); omit for a fresh adapter")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--lora-layers", type=int, default=24)
    ap.add_argument("--model", default=str(HERE.parent / "models/MiniCPM5-1B"))
    ap.add_argument("--solutions", nargs="+", default=[str(env.DATA_DIR / "solutions.jsonl")])
    ap.add_argument("--replay", nargs="*", default=[], help="run directories (under runs/) to replay correct transcripts from")
    ap.add_argument("--replay-ratio", type=float, default=1.0, help="student transcripts per teacher example")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32, help="examples per update")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-nll", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.9)
    ap.add_argument("--micro-tokens", type=int, default=8192)
    ap.add_argument("--mem-gb", type=float, default=30)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = HERE / "runs" / args.name
    if run.exists():
        raise FileExistsError(f"Run already exists: {run}")
    log = train.RunLog(run)
    set_memory_budget(args.mem_gb)
    env.ensure_databases()
    evolve.load_accepted()
    model, tok = load(args.model)
    model.freeze()
    if args.resume:
        state = json.loads((Path(args.resume) / "trainer_state.json").read_text())
        config = json.loads((Path(args.resume) / "adapter_config.json").read_text())
        model = load_adapters(model, args.resume)
        evaluated = dict(train.tree_flatten(model.trainable_parameters()))
        if (Path(args.resume) / "raw_adapters.safetensors").exists():
            model.load_weights(str(Path(args.resume) / "raw_adapters.safetensors"), strict=False)
    else:
        lora = {"rank": args.rank, "scale": args.scale, "dropout": 0.0}
        train.linear_to_lora_layers(model, args.lora_layers, lora)
        state = {"step": 0, "curriculum": {"rate": {}, "seen": {}}}
        config = {"num_layers": args.lora_layers, "lora_parameters": lora}
        evaluated = None
    model.eval()
    average = train.WeightAverage(model, args.ema_decay, evaluated)
    opt = optim.Adam(learning_rate=args.lr)
    if args.resume:
        train.load_optimizer(opt, model, Path(args.resume), args.lr)
    eval_args = SimpleNamespace(eval_per_template=2, eval_k=4, eval_heldout_questions=4, eval_heldout_k=8,
                                temp=1.0, max_turns=5, max_tokens=args.max_tokens, hide_schema=False, gen_batch=128)
    log.write("config.json", {**vars(args), "started": time.time(), "kind": "sft", "start_step": state["step"],
                             "verifier_version": env.VERIFIER_VERSION, "eval_version": env.EVAL_VERSION,
                             "git": train.git_revision(), "steps": args.epochs, "prompts": args.batch, "group": 1,
                             "rank": config["lora_parameters"]["rank"], "temp": 1.0, "max_turns": 5, "tiers": "1,2,3",
                             "hide_schema": False, "teacher": None})

    teacher = [ex for path in args.solutions for ex in teacher_examples(tok, path, args.max_tokens)]
    student = replay_examples(tok, args.replay, int(len(teacher) * args.replay_ratio), args.seed)
    examples = teacher + student
    nll = [train.action_nll(model, t, m) for _, _, t, m in examples]
    kept = [ex for ex, v in zip(examples, nll) if v <= args.max_nll]
    summary = {"teacher": len(teacher), "student": len(student), "kept": len(kept),
               "kept_teacher": sum(ex[0] == "teacher" for ex in kept),
               "templates": len({ex[1] for ex in kept}), "nll_median": sorted(nll)[len(nll) // 2] if nll else None}
    print(summary)

    with average.applied(model):
        before = train.run_eval(model, tok, eval_args, log, state["step"])
        train.run_eval(model, tok, eval_args, log, state["step"], check=f"before_sft_step_{state['step']:05d}")
    rng = random.Random(args.seed)
    updates = 0
    for epoch in range(args.epochs):
        rng.shuffle(kept)
        for i in range(0, len(kept), args.batch):
            batch = [(t, m, 1.0) for _, _, t, m in kept[i:i + args.batch]]
            log.status(phase="train", step=updates + 1)
            loss, grad_norm, n_tok = train.train_step(model, opt, batch, 1.0, args.micro_tokens, 1.0)
            average.update(model)
            updates += 1
            log.append("metrics.jsonl", {"step": updates, "time": time.time(), "epoch": epoch, "loss": loss,
                                         "grad_norm": grad_norm, "train_tokens": n_tok, "reward_mean": -loss,
                                         "outcomes": {}, "by_template": {}, "by_tier": {}, "turns_mean": 0,
                                         "literal_frac": 0, "literal_correct_frac": 0, "gen_tokens_mean": 0,
                                         "groups_kept": len(batch), "groups_total": len(batch), "rollout_s": 0,
                                         "train_s": 0, "gen_tps": 0, "peak_mem_gb": mx.get_peak_memory() / 1024**3})
            print(f"epoch {epoch} update {updates} loss {loss:.4f} |g| {grad_norm:.3f}")
    curriculum = Curriculum()
    curriculum.rate, curriculum.seen = state["curriculum"]["rate"], train.defaultdict(int, state["curriculum"]["seen"])
    ckpt_args = SimpleNamespace(lora_layers=config["num_layers"], rank=config["lora_parameters"]["rank"],
                                scale=config["lora_parameters"]["scale"], name=args.name)
    train.save_checkpoint(model, opt, run / "final", ckpt_args, state["step"], curriculum, None, average)
    with average.applied(model):
        after = train.run_eval(model, tok, eval_args, log, state["step"] + 1)
        train.run_eval(model, tok, eval_args, log, state["step"] + 1, check=f"after_sft_step_{state['step']:05d}")
    result = {"summary": summary, "updates": updates, "before": before["by_split"], "after": after["by_split"]}
    log.write("sft_result.json", result)
    log.status(phase="done", step=updates, stop_reason="completed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
