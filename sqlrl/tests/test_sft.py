import json

import env
import sft


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


def test_episode_masks_cover_only_assistant_turns(monkeypatch):
    monkeypatch.setattr(sft, 'prompt_tokens', lambda tok, task, show: [1, 2, 3])
    tokens, mask = sft.episode_tokens(Tokenizer(), 'q', [('run', 'rows'), ('submit', 'final')])
    assert len(tokens) == len(mask) and mask[:3] == [0, 0, 0]
    assert sum(mask) == len(b'run<|im_end|>') + len(b'submit<|im_end|>')
    assert b'final' not in bytes(t for t in tokens if t < 256)  # no observation after the submission


def test_teacher_examples_skip_heldout_and_eval_questions(monkeypatch, tmp_path):
    monkeypatch.setattr(sft, 'prompt_tokens', lambda tok, task, show: [1])
    held = env.HELDOUT_TEST[0]
    rows = [{'template': held, 'question': 'x', 'actions': [['submit_sql', 'SELECT 1', '1']]},
            {'template': 'count_city', 'question': next(iter(env.eval_question_texts())),
             'actions': [['submit_sql', 'SELECT 1', '1']]},
            {'template': 'count_city', 'question': 'How many customers live in Leeds?',
             'actions': [['submit_sql', "SELECT COUNT(*) FROM customers WHERE city = 'Leeds'", '9']]}]
    path = tmp_path / 's.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    out = sft.teacher_examples(Tokenizer(), path, 256)
    assert [ex[1] for ex in out] == ['count_city'] and out[0][0] == 'teacher'
