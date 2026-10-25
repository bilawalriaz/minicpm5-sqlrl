"""Batched multi-turn rollouts of the SQL agent with mlx-lm.

Each episode is kept as one flat token sequence plus a mask marking which
tokens the policy generated. Tool results are spliced in as tokens with
mask 0, so the trainer can score the whole trajectory in one forward pass and
only push gradient through the model's own actions.

Prefill is most of the cost of a rollout, so nothing is prefilled twice: the
system prompt every episode shares is processed once, and each episode keeps
its KV cache between turns, so a new turn only feeds the tool result.
"""

import time
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

import env

# Text the chat template would put between an assistant turn ending in
# <|im_end|> and the next assistant turn, when the user turn is a tool result.
# The empty think block keeps the model in no-think mode on every turn.
OBS_TEMPLATE = "\n<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_budget = None
# Each in-flight episode holds its KV cache between turns (~24KB per token), so
# bigger rollouts run in chunks of this many episodes.
MAX_IN_FLIGHT = 128


def set_memory_budget(gb):
    """Keep MLX inside `gb` of unified memory.

    MLX's defaults on a 36GB Mac allow 1.5x the recommended working set (45GB)
    and keep freed buffers cached up to that limit, so repeated rollouts grow
    until macOS swaps. Cap the total, keep the free-buffer cache small, and
    never wire (pin) more than the budget or than macOS allows (~28GiB here).
    """
    global _budget
    budget = _budget = int(gb * 1024**3)
    mx.set_memory_limit(budget)
    mx.set_cache_limit(min(budget // 8, 2 * 1024**3))
    mx.set_wired_limit(_wired_cap())
    return budget


def _wired_cap():
    return min(_budget, mx.device_info()["max_recommended_working_set_size"])


@dataclass
class Episode:
    task: env.Task
    tokens: list
    mask: list
    prompt_len: int
    turns: int = 0
    done: bool = False
    reward: float = 0.0
    outcome: str = ""
    # Human-readable turns for logs: {"role": "assistant"|"tool", "text": ...}
    segments: list = field(default_factory=list)
    final_sql: str = None
    answer_rows: list = None
    gold_rows: list = None
    source: str = "student"  # "teacher" for a rescued group's injected solution
    verifier_reason: str = ""
    prose_tokens: int = 0  # tokens of text outside function calls, across all turns
    sample_logprobs: list = field(default_factory=list)  # sampler log-probability of each generated token
    cache: list = field(default=None, repr=False)


def prompt_tokens(tokenizer, task, show_schema=True):
    messages = [
        {"role": "system", "content": env.system_prompt(show_schema)},
        {"role": "user", "content": task.question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tools=env.TOOLS, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )
    return tokenizer.encode(text, add_special_tokens=False)


def _shared_prefix_cache(model, prompts):
    """Prefill the longest prefix all prompts share (the system prompt) once."""
    first = prompts[0]
    n = min(len(p) for p in prompts) - 1  # leave at least one token to feed per prompt
    for p in prompts[1:]:
        while n and p[:n] != first[:n]:
            n -= 1
    cache = make_prompt_cache(model)
    if n:
        model(mx.array(first[:n])[None], cache=cache)
        mx.eval([c.state for c in cache])
    return cache, n


def _generate(model, tokenizer, prompts, caches, temp, max_tokens, batch_size, on_progress=None):
    """Sample one assistant turn for every (new tokens, cache) pair.

    The batch generator copies the caches it is given into its own batch, so
    one shared prefix cache can seed every prompt. Returns [(tokens,
    finish_reason, cache)], where cache already holds every generated token.
    on_progress, if given, is called about once a second with the tokens
    generated so far for each prompt, in prompt order."""
    gen = BatchGenerator(
        model,
        max_tokens=max_tokens,
        stop_tokens=tokenizer.eos_token_ids,
        sampler=make_sampler(temp=temp),
        completion_batch_size=batch_size,
        prefill_batch_size=min(8, batch_size),
        prefill_step_size=512,
    )
    if _budget:
        # BatchGenerator wires the device's whole recommended working set; hold it to our budget.
        mx.set_wired_limit(_wired_cap())
    uids = gen.insert(prompts, caches=caches)
    out = {u: [] for u in uids}
    logprobs = {u: [] for u in uids}
    done = {}
    n_tokens = 0
    last_report = time.time()
    try:
        while responses := gen.next():
            # One host transfer per decoding step, for training/sampling consistency checks.
            for r, lp in zip(responses, mx.stack([r.logprobs[r.token] for r in responses]).tolist()):
                logprobs[r.uid].append(lp)
            for r in responses:
                out[r.uid].append(int(r.token))
                n_tokens += 1
                if r.finish_reason is not None:
                    # extract() returns a lazy slice of the whole batch's KV buffer;
                    # evaluate it now or every finished sequence pins that buffer.
                    mx.eval([c.state for c in r.prompt_cache])
                    done[r.uid] = (r.finish_reason, r.prompt_cache)
            if _budget and mx.get_active_memory() > _budget:
                raise MemoryError(
                    f"MLX is using {mx.get_active_memory() / 1024**3:.1f}GB, over the {_budget / 1024**3:.0f}GB budget. "
                    "Lower --gen-batch or --max-tokens, or raise --mem-gb."
                )
            if on_progress and time.time() - last_report >= 1.0:
                on_progress([out[u] for u in uids])
                last_report = time.time()
    finally:
        gen.close()
    mx.clear_cache()
    return [(out[u], *done[u], logprobs[u]) for u in uids], n_tokens


def rollout(model, tokenizer, tasks, temp=1.0, max_turns=5, max_tokens=256,
            show_schema=True, batch_size=32, verbose=False, on_turn=None, live=None, prose_penalty=0.0,
            no_answer_reward=-0.2):
    """live, if given, is called about once a second with (episodes, {id(episode):
    text of the turn being generated}) so a dashboard can show rollouts as they stream.

    prose_penalty: reward lost per token of non-SQL text in a correct episode (apologies,
    narration), capped at 0.3 so correctness always dominates. Within a group, concise
    correct answers then beat chatty ones.

    no_answer_reward: penalty for plain-text termination, token truncation, or
    exhausting the turn budget without submitting SQL."""
    if len(tasks) > MAX_IN_FLIGHT:
        episodes, stats = [], {"gen_tokens": 0, "seconds": 0.0}
        for i in range(0, len(tasks), MAX_IN_FLIGHT):
            eps, s = rollout(model, tokenizer, tasks[i:i + MAX_IN_FLIGHT], temp, max_turns, max_tokens,
                             show_schema, batch_size, verbose, on_turn, live, prose_penalty, no_answer_reward)
            episodes += eps
            stats = {k: stats[k] + s[k] for k in stats}
        return episodes, stats
    if not tasks:
        raise ValueError("Rollout needs at least one task")
    if len({str(t.db_path) for t in tasks}) != 1:
        raise ValueError("A rollout batch must use one database")
    cons = env.verifier_connections(tasks[0].db_path)
    try:
        con = cons[0]
        gold = {}
        episodes = []
        for t in tasks:
            p = prompt_tokens(tokenizer, t, show_schema)
            episodes.append(Episode(t, list(p), [0] * len(p), len(p)))
        prefix_cache, prefix_len = _shared_prefix_cache(model, [e.tokens for e in episodes])
        fed = {id(e): prefix_len for e in episodes}  # tokens already in each episode's cache

        t0, gen_tokens = time.time(), 0
        for turn in range(max_turns):
            active = [e for e in episodes if not e.done]
            if not active:
                break
            def report(partials, active=active):
                stop = tokenizer.eos_token_ids
                live(episodes, {id(e): tokenizer.decode([t for t in p if t not in stop]) for e, p in zip(active, partials)})

            results, n = _generate(
                model, tokenizer,
                [e.tokens[fed[id(e)]:] for e in active],
                [e.cache or prefix_cache for e in active],
                temp, max_tokens, batch_size, report if live else None,
            )
            gen_tokens += n
            for e, (toks, reason, cache, lps) in zip(active, results):
                e.tokens += toks
                e.sample_logprobs += lps
                e.mask += [1] * len(toks)
                e.turns += 1
                e.cache, fed[id(e)] = cache, len(e.tokens)
                text = tokenizer.decode(toks[:-1] if reason == "stop" else toks)
                e.segments.append({"role": "assistant", "text": text})
                e.prose_tokens += len(tokenizer.encode(env.prose(text), add_special_tokens=False))
                if reason != "stop":
                    e.done, e.reward, e.outcome = True, no_answer_reward, "length"
                    continue
                try:
                    name, query = env.parse_call(text)
                    if name is None:
                        e.done, e.reward, e.outcome = True, no_answer_reward, "no_query"
                        e.verifier_reason = "No submit_sql call"
                        continue
                    if name not in ("run_sql", "submit_sql"):
                        raise ValueError(f"Unknown function {name}")
                    rows, err = env.execute(con, query)
                except ValueError as exc:
                    name, rows, err = None, None, str(exc)
                result = env.format_rows(rows, err)
                e.segments.append({"role": "tool", "text": result})
                if name == "submit_sql":
                    e.done, e.final_sql = True, query
                    if err:
                        e.reward, e.outcome, e.verifier_reason = 0.0, "wrong", err
                        continue
                    if e.task.gold_sql not in gold:
                        expected, gold_err = env.execute(con, e.task.gold_sql)
                        if gold_err:
                            raise ValueError(f"Reference query failed: {gold_err}")
                        gold[e.task.gold_sql] = expected
                    e.answer_rows, e.gold_rows = rows, gold[e.task.gold_sql]
                    ok, e.verifier_reason = env.grade(cons, query, e.task.gold_sql)
                    e.reward, e.outcome = (1.0 - min(0.3, prose_penalty * e.prose_tokens), "correct") if ok else (0.0, "wrong")
                    continue
                if e.turns >= max_turns:
                    e.done, e.reward, e.outcome = True, no_answer_reward, "max_turns"
                    continue
                obs = tokenizer.encode(OBS_TEMPLATE.format(result), add_special_tokens=False)
                e.tokens += obs
                e.mask += [0] * len(obs)
            for e in active:
                if e.done:
                    e.cache = None
            if live:
                live(episodes, {})
            if on_turn:
                on_turn(turn + 1, len(active), gen_tokens, time.time() - t0)
            if verbose:
                print(f"  turn {turn + 1}: {len(active)} active, {gen_tokens} tokens generated, "
                      f"{time.time() - t0:.0f}s, {mx.get_active_memory() / 1024**3:.1f}GB active")

        for e in episodes:
            e.cache = None
            if not e.done:
                e.done, e.reward, e.outcome = True, no_answer_reward, "max_turns"
        if live:
            live(episodes, {})
        return episodes, {"gen_tokens": gen_tokens, "seconds": time.time() - t0}
    finally:
        for connection in cons:
            connection.close()


def transcript(tokenizer, e):
    return tokenizer.decode(e.tokens[e.prompt_len:])
