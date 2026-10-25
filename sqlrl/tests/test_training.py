from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

import train


def test_eval_uses_binary_outcomes_for_every_accuracy(monkeypatch):
    task = SimpleNamespace(template='example', tier=1, question='q')
    eps = [SimpleNamespace(task=task, reward=1.0, outcome='correct'),
           SimpleNamespace(task=task, reward=-0.2, outcome='length')]
    monkeypatch.setattr(train, 'rollout', lambda *a, **k: (eps, {'seconds': 1}))
    monkeypatch.setattr(train, 'episode_record', lambda *a: {})
    log = SimpleNamespace(status=lambda **kw: None, write=lambda *a: None,
                          append=lambda name, row: rows.append(row))
    rows = []
    args = SimpleNamespace(eval_per_template=1, eval_k=1, temp=1.0, max_turns=5,
                           max_tokens=256, hide_schema=False, gen_batch=1)
    train.run_eval(SimpleNamespace(eval=lambda: None), None, args, log, 0)
    row = rows[0]
    assert row['pass1'] == row['by_tier']['1'] == row['by_template']['example'] == 0.5


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Embedding(32, 8)
        self.lm_head = nn.Linear(8, 32)


def test_gradient_accumulation_matches_different_padding_and_microbatches():
    mx.random.seed(7)
    model = TinyModel()
    batch = [([1, 2, 3, 4], [0, 0, 1, 1], 0.75),
             ([1, 5, 6, 7, 8, 9], [0, 0, 1, 0, 1, 1], -0.25)]
    function = nn.value_and_grad(model, train.seq_logprob_loss)
    small, loss_small = train._accumulate(model, function, batch, 1.0, 4, 5)
    large, loss_large = train._accumulate(model, function, batch, 1.0, 100, 5)
    assert loss_small == pytest.approx(loss_large, abs=1e-6)
    assert mx.allclose(small['model']['weight'], large['model']['weight'], atol=1e-6).item()
    assert mx.allclose(small['lm_head']['weight'], large['lm_head']['weight'], atol=1e-6).item()
