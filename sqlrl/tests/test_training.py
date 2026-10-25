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
    args = SimpleNamespace(eval_per_template=1, eval_k=1, temp=1.0, max_turns=5, eval_heldout_questions=1,
                           eval_heldout_k=1, max_tokens=256, hide_schema=False, gen_batch=1)
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
    train.load_optimizer(restored, model, tmp_path / 'step_00001', 0.02)
    for (k, a), (_, b) in zip(sorted(tree_flatten(opt.state)), sorted(tree_flatten(restored.state))):
        if k != 'learning_rate':
            assert mx.array_equal(a, b).item(), k
    assert restored.learning_rate.item() == pytest.approx(0.02)  # --lr wins over the saved rate
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


def test_selection_smooths_and_stops_without_improvement():
    sel = train.Selection(patience=2, drop=0.15, drop_evals=0)
    assert sel.add(0, 0.9, candidate=False) is None  # the untrained start is never selected
    for step, score in [(10, 0.4), (20, 0.6), (30, 0.8), (40, 0.5)]:
        assert sel.add(step, score) is None
    # Smoothed: 20 -> 0.5, 30 -> 0.7, 40 -> 0.65. A single high score at 10 does not win.
    assert sel.best() == (30, pytest.approx(0.7))
    assert 'no selection improvement in 2' in sel.add(50, 0.6)


def test_selection_stops_after_sustained_drop():
    sel = train.Selection(patience=0, drop=0.15, drop_evals=2)
    for step, score in [(10, 0.7), (20, 0.5)]:
        assert sel.add(step, score) is None
    assert 'below its best for 2' in sel.add(30, 0.5)


def test_burst_due_on_sustained_low_signal_with_cooldown():
    assert not train.burst_due([2, 2, 2, 2], 4, None, 5, 6, busy=False)        # window not full
    assert train.burst_due([9, 2, 2, 2, 2, 2], 6, None, 5, 6, busy=False)
    assert not train.burst_due([2] * 6, 6, None, 5, 6, busy=True)               # generation already running
    assert not train.burst_due([2] * 8, 8, 6, 5, 6, busy=False)                 # cooldown
    assert train.burst_due([2] * 11, 11, 6, 5, 6, busy=False)
    assert not train.burst_due([7] * 6, 6, None, 5, 6, busy=False)
    assert not train.burst_due([2] * 6, 6, None, 5, 0, busy=False)              # disabled


def test_linear_learning_rate_decay():
    assert train.learning_rate(5, 1, 9, 3e-5, None) == 3e-5
    assert train.learning_rate(1, 1, 9, 3e-5, 1e-5) == pytest.approx(3e-5)
    assert train.learning_rate(5, 1, 9, 3e-5, 1e-5) == pytest.approx(2e-5)
    assert train.learning_rate(9, 1, 9, 3e-5, 1e-5) == pytest.approx(1e-5)


def ep(outcome, reward, source="student"):
    return SimpleNamespace(outcome=outcome, reward=reward, source=source)


def test_wrong_answers_never_get_positive_advantage():
    # All students failed: no student signal; the teacher's solution is the only positive.
    rescued = [ep("wrong", 0.0), ep("length", -0.2), ep("wrong", 0.0), ep("correct", 1.0, "teacher")]
    adv = train.group_advantages(rescued)
    assert adv[:3] == [0.0, 0.0, 0.0] and adv[3] > 0
    assert train.group_advantages([ep("wrong", 0.0), ep("max_turns", -0.2)]) == [0.0, 0.0]
    # One correct, one wrong, many unanswered: the wrong answer would be above the mean.
    group = [ep("correct", 1.0), ep("wrong", 0.0)] + [ep("length", -0.2)] * 6
    adv = train.group_advantages(group)
    assert adv[0] > 0 and adv[1] == 0.0 and all(a < 0 for a in adv[2:])
    assert abs(sum(train.group_advantages([ep("correct", 1.0), ep("wrong", 0.0)]))) < 1e-9
