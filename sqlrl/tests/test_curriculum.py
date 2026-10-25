from types import SimpleNamespace

import env
from curriculum import Curriculum


def episodes(template, solved, n=40):
    task = SimpleNamespace(template=template)
    return [SimpleNamespace(task=task, source="student", outcome="correct" if i < solved else "wrong") for i in range(n)]


def test_weights_exclude_heldout_and_mastered_templates_drop_to_replay():
    c = Curriculum(alpha=1.0)
    weights = c.weights()
    assert not set(weights) & set(env.HELDOUT_TEMPLATES)
    assert abs(sum(v["share"] for v in c.snapshot()["templates"].values()) - 1) < 1e-9
    c.update(episodes("count_city", 40))
    c.update(episodes("churned", 20))
    assert c.weight("count_city") == c.replay
    assert c.weight("churned") == c.floor + 0.25


def test_evolve_targets_continue_after_every_skill_is_learned():
    c = Curriculum(alpha=1.0)
    for name in env.BUILTIN:
        if name not in env.HELDOUT_TEMPLATES:
            c.update(episodes(name, 40))
    assert c.weakest_skills(2) == []
    targets = c.evolve_targets(2)
    assert len(targets) == 2
    count = lambda s: sum(s in env.SKILLS[t] for t in env.TEMPLATES if t not in env.HELDOUT_TEMPLATES)
    assert max(count(s) for s in targets) <= min(count(s) for s in env.SKILL_DESCRIPTIONS if s not in targets)


def test_replay_weights_cover_only_mastered_templates():
    c = Curriculum(alpha=1.0)
    assert c.replay_weights() is None
    c.update(episodes("count_city", 40))
    c.update(episodes("churned", 20))
    replay = c.replay_weights()
    assert replay["count_city"] == 1.0 and replay["churned"] == 0.0
    tasks = env.sample_tasks(8, seed=1, weights=replay)
    assert {t.template for t in tasks} == {"count_city"}


def test_failing_templates_need_evidence():
    c = Curriculum(alpha=1.0)
    c.update(episodes("churned", 0, n=8))
    assert c.failing(2) == []
    c.update(episodes("churned", 0, n=16))
    c.update(episodes("count_city", 40))
    assert c.failing(2) == ["churned"]
