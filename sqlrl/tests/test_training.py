import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

import train


def test_eval_uses_binary_outcomes_for_every_accuracy(monkeypatch):
    task = SimpleNamespace(template='example', tier=1, question='q', split='reserved_params')
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
    assert row['pass1'] == row['by_tier']['1'] == row['by_template']['example'] == row['by_split']['reserved_params'] == 0.5


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


def test_checkpoint_restores_optimizer_state_and_refuses_overwrite(tmp_path):
    from curriculum import Curriculum
    mx.random.seed(3)
    model = TinyModel()
    opt = optim.Adam(learning_rate=0.1)
    batch = [([1, 2, 3, 4], [0, 0, 1, 1], 0.5)]
    grads = nn.value_and_grad(model, train.seq_logprob_loss)
    g, _ = train._accumulate(model, grads, batch, 1.0, 100, 2)
    opt.update(model, g)
    curriculum = Curriculum()
    curriculum.rate['count_city'] = 0.25
    args = SimpleNamespace(lora_layers=1, rank=2, scale=1.0, name='t')
    train.save_checkpoint(model, opt, tmp_path / 'step_00001', args, 1, curriculum)
    assert not (tmp_path / 'step_00001.partial').exists()
    restored = optim.Adam(learning_rate=0.1)
    train.load_optimizer(restored, model, tmp_path / 'step_00001')
    for (k, a), (_, b) in zip(sorted(tree_flatten(opt.state)), sorted(tree_flatten(restored.state))):
        assert mx.array_equal(a, b).item(), k
    state = json.loads((tmp_path / 'step_00001' / 'trainer_state.json').read_text())
    assert state['step'] == 1 and state['curriculum']['rate'] == {'count_city': 0.25}
    with pytest.raises(FileExistsError):
        train.save_checkpoint(model, opt, tmp_path / 'step_00001', args, 1, curriculum)


def test_nonfinite_gradients_leave_weights_unchanged():
    model = TinyModel()
    model.lm_head.weight = model.lm_head.weight * float('nan')
    before = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
    opt = optim.Adam(learning_rate=0.1)
    train.grad_checkpoint = lambda layer: None
    model.layers = [model.lm_head]
    with pytest.raises(FloatingPointError):
        train.train_step(model, opt, [([1, 2, 3], [0, 1, 1], 1.0)], 1.0, 100, 1.0)
    after = dict(tree_flatten(model.parameters()))
    assert all(mx.array_equal(before[k], after[k], equal_nan=True).item() for k in before)


def test_signal_during_update_is_deferred_until_it_finishes():
    signals = train.Signals.__new__(train.Signals)
    signals.critical, signals.pending = False, None
    finished = []
    with pytest.raises(train.Stop):
        with signals:
            signals.handle(15, None)
            finished.append(True)
    assert finished == [True]
    with pytest.raises(train.Stop):
        signals.handle(15, None)
