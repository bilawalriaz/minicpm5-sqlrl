from types import SimpleNamespace

import pytest

import env
import rollout as engine
from teacher import as_episode


class Tokenizer:
    eos_token_ids = {0}

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()


def call(name='submit_sql', query='SELECT COUNT(*) FROM orders'):
    return f'<function name="{name}"><param name="query">{query}</param></function>'


def run(monkeypatch, turns, *, max_turns=5):
    tok = Tokenizer()
    pending = iter(turns)
    monkeypatch.setattr(engine, 'prompt_tokens', lambda *a: [100, 101])
    monkeypatch.setattr(engine, '_shared_prefix_cache', lambda *a: ([], 1))

    def generate(*args, **kwargs):
        text, reason = next(pending)
        tokens = tok.encode(text) + ([0] if reason == 'stop' else [])
        return [(tokens, reason, [], [0.0] * len(tokens))], len(tokens)

    monkeypatch.setattr(engine, '_generate', generate)
    task = env.Task('count', 1, 'How many orders?', 'SELECT COUNT(*) FROM orders')
    eps, _ = engine.rollout(None, tok, [task], max_turns=max_turns, no_answer_reward=-0.2)
    return eps[0]


def test_submit_on_final_turn_is_accepted(monkeypatch):
    ep = run(monkeypatch, [(call(), 'stop')], max_turns=1)
    assert ep.outcome == 'correct' and ep.reward == 1
    assert ep.answer_rows == [(1200,)]
    assert len(ep.tokens) == len(ep.mask)
    assert len(ep.sample_logprobs) == sum(ep.mask)
    assert not any(ep.mask[:ep.prompt_len])


@pytest.mark.parametrize('turns,outcome', [
    ([(call('run_sql'), 'stop'), ('Here is a wrong answer', 'stop')], 'no_query'),
    ([(call(), 'length')], 'length'),
    ([(call('run_sql'), 'stop')], 'max_turns'),
])
def test_unfinished_episodes_get_same_penalty(monkeypatch, turns, outcome):
    ep = run(monkeypatch, turns, max_turns=len(turns))
    assert ep.outcome == outcome and ep.reward == -0.2
    assert ep.final_sql is None


def test_bad_submission_does_not_fall_back_to_previous_success(monkeypatch):
    ep = run(monkeypatch, [(call('run_sql'), 'stop'), (call(query='SELECT nonexistent FROM orders'), 'stop')])
    assert ep.outcome == 'wrong' and ep.reward == 0
    assert 'nonexistent' in ep.final_sql


def test_malformed_or_unknown_calls_can_be_repaired(monkeypatch):
    for bad in ['<function name="run_sql">', call('unknown'), call() + call()]:
        ep = run(monkeypatch, [(bad, 'stop'), (call(), 'stop')])
        assert ep.outcome == 'correct'
        assert ep.segments[1]['text'].startswith('Error:')
        assert 0 in ep.mask[ep.prompt_len:]


def test_complete_answer_is_saved_not_only_preview(monkeypatch):
    query = 'SELECT id FROM orders'
    task = env.Task('all', 1, 'All order IDs', query)
    original = env.Task
    # run() constructs a count task; give it this list-returning reference.
    monkeypatch.setattr(env, 'Task', lambda *a: task)
    ep = run(monkeypatch, [(call(query=query), 'stop')])
    assert ep.outcome == 'correct' and len(ep.answer_rows) == 1200
    assert '1200 rows total' in ep.segments[-1]['text']
    monkeypatch.setattr(env, 'Task', original)


def test_teacher_uses_same_terminal_call_and_masks():
    task = env.Task('count', 1, 'How many orders?', 'SELECT COUNT(*) FROM orders')
    student = SimpleNamespace(tokens=[100, 101], prompt_len=2, task=task)
    solution = {'actions': [('run_sql', 'SELECT name FROM products LIMIT 1', 'Red Ball'),
                             ('submit_sql', task.gold_sql, '1200')], 'final_sql': task.gold_sql}
    ep = as_episode(Tokenizer(), student, solution, max_tokens=1000)
    assert ep.turns == 2 and ep.source == 'teacher'
    assert env.parse_call(ep.segments[-2]['text'])[0] == 'submit_sql'
    assert len(ep.tokens) == len(ep.mask) and not any(ep.mask[:2])
    assert 0 in ep.mask[2:]
    with pytest.raises(ValueError, match='token limit'):
        as_episode(Tokenizer(), student, solution, max_tokens=10)
