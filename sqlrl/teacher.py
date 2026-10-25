"""Verified API-teacher SQL submissions and student-format trajectory injection.

The teacher uses run_sql and submit_sql with the student's schema visibility and
turn budget. Credentials are read from sqlrl/.env or the environment.
"""

import json
import os
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

import env

BASE_URL = "https://opencode.ai/zen/go/v1"


def _api_key():
    # sqlrl/.env wins over the environment: a stale key exported by a shell
    # profile otherwise shadows it and every call fails with api_key_expired.
    dotenv = Path(__file__).parent / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENCODE_GO_API_KEY":
                return value.strip()
    if key := os.environ.get("OPENCODE_GO_API_KEY"):
        return key
    raise RuntimeError("Put OPENCODE_GO_API_KEY in sqlrl/.env or the environment")


def make_client(timeout=120):
    # The Go gateway asks clients to name themselves and send one session ID per conversation.
    return OpenAI(base_url=BASE_URL, api_key=_api_key(), timeout=timeout, max_retries=2,
                  default_headers={"User-Agent": "sqlrl-teacher/0.1"})


def session_header():
    return {"x-opencode-session": f"sqlrl-{uuid.uuid4()}"}


def cost_of(resp):
    """Dollars a response cost against the Go allowance."""
    return (resp.usage.model_extra.get("opencode") or {}).get("list_cost", 0.0)


class Teacher:
    def __init__(self, model, max_turns=5, workers=8, effort="low", timeout=120, show_schema=True):
        self.lock = threading.Lock()
        self.show_schema = show_schema
        self.model = model
        self.max_turns = max_turns
        self.effort = effort
        self.client = make_client(timeout)
        self.pool = ThreadPoolExecutor(workers)
        # Templates repeat with the same parameters, so the same question comes back often.
        self.solved = {}
        self.spent = 0.0  # dollars against the Go allowance, this run

    def submit(self, tasks):
        """Start solving tasks in the background. Returns {question: Future}."""
        return {t.question: self.pool.submit(self.solve, t) for t in {t.question: t for t in tasks}.values()}

    def solve(self, task):
        key = (str(task.db_path), task.question, task.gold_sql, self.show_schema)
        with self.lock:
            if key in self.solved:
                return self.solved[key]
        cons = env.verifier_connections(task.db_path)
        try:
            messages = [
                {"role": "system", "content": env.system_prompt(self.show_schema)},
                {"role": "user", "content": task.question},
            ]
            actions, final_sql, correct = [], None, False
            t0, usage, cost = time.time(), 0, 0.0
            session = session_header()
            for _ in range(self.max_turns):
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=env.TOOLS, temperature=0.2, max_tokens=1024,
                    reasoning_effort=self.effort, extra_headers=session,
                )
                usage += resp.usage.total_tokens
                cost += cost_of(resp)
                msg = resp.choices[0].message
                calls = msg.tool_calls or []
                if resp.choices[0].finish_reason != "tool_calls" or len(calls) != 1:
                    break
                call = calls[0]
                name = call.function.name
                if name not in ("run_sql", "submit_sql"):
                    break
                try:
                    params = json.loads(call.function.arguments or "{}")
                    if not isinstance(params, dict) or set(params) != {"query"} or not isinstance(params["query"], str):
                        break
                    query = params["query"].strip()
                except (ValueError, TypeError):
                    break
                rows, err = env.execute(cons[0], query)
                result = env.format_rows(rows, err)
                actions.append((name, query, result))
                if name == "submit_sql":
                    final_sql = query
                    if err is None:
                        correct, _ = env.grade(cons, query, task.gold_sql)
                    break
                messages.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": [{"id": call.id, "type": "function",
                                                 "function": {"name": name, "arguments": call.function.arguments}}]})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            out = {
                "correct": correct, "actions": actions, "final_text": "", "final_sql": final_sql,
                "seconds": time.time() - t0, "tokens": usage, "cost": cost, "model": self.model,
            }
            with self.lock:
                self.spent += cost
                if correct:
                    self.solved[key] = out
            return out
        finally:
            for con in cons:
                con.close()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.client.close()


def as_episode(tokenizer, student_ep, solution, max_tokens=256):
    """Render the teacher using the student's exact terminal submission protocol."""
    from rollout import OBS_TEMPLATE, Episode

    prompt = student_ep.tokens[:student_ep.prompt_len]
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    tokens, mask = list(prompt), [0] * len(prompt)
    segments = []
    for name, query, result in solution["actions"]:
        value = f"<![CDATA[{query}]]>" if any(c in query for c in "<&\n") else query
        text = f'<function name="{name}"><param name="query">{value}</param></function>'
        ids = tokenizer.encode(text, add_special_tokens=False) + im_end
        if len(ids) > max_tokens:
            raise ValueError("Teacher action exceeds student's per-turn token limit")
        tokens.extend(ids)
        mask.extend([1] * len(ids))
        segments.append({"role": "assistant", "text": text})
        segments.append({"role": "tool", "text": result})
        if name != "submit_sql":
            obs = tokenizer.encode(OBS_TEMPLATE.format(result), add_special_tokens=False)
            tokens.extend(obs)
            mask.extend([0] * len(obs))
    if not solution["actions"] or solution["actions"][-1][0] != "submit_sql":
        raise ValueError("Teacher did not submit SQL")
    return Episode(
        student_ep.task, tokens, mask, len(prompt), turns=len(solution["actions"]), done=True,
        reward=1.0, outcome="correct", segments=segments, final_sql=solution["final_sql"], source="teacher",
        verifier_reason="Verified teacher submission",
    )
