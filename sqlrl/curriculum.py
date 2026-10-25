"""Adaptive curriculum: track how often the student solves each template and
sample the next batch toward what it can just about do.

GRPO only learns from groups with mixed rewards, so a template solved 0% or
100% of the time is wasted compute. The sampling weight is floor + p(1 - p),
which peaks at a 50% solve rate. The floor keeps mastered templates in rotation
so they are not forgotten. With a teacher, near-0% templates still produce
signal (the teacher's solution rescues the group), so they get a bonus.
A template solved at least `mastered` of the time over `mastered_seen` rollouts drops
to the smaller `replay` weight, so most of each batch can still carry signal. Mastered
templates come back through a fixed replay share of each batch (see train.py), because
without it the model forgot simple questions within four steps of meeting many harder ones.
Held-out evaluation templates get no weight.
"""

from collections import defaultdict

import env


class Curriculum:
    def __init__(self, alpha=0.3, floor=0.05, prior=0.5, hard_bonus=0.0, replay=0.01, mastered=0.95, mastered_seen=32):
        self.alpha = alpha            # weight of the newest step in the moving average
        self.floor = floor
        self.prior = prior            # assumed solve rate before a template is seen
        self.hard_bonus = hard_bonus  # extra weight below a 10% solve rate (use with a teacher)
        self.replay, self.mastered, self.mastered_seen = replay, mastered, mastered_seen
        self.rate = {}
        self.seen = defaultdict(int)

    def update(self, episodes):
        solved = defaultdict(list)
        for e in episodes:
            if e.source == "student":
                solved[e.task.template].append(e.outcome == "correct")
        for t, xs in solved.items():
            p = sum(xs) / len(xs)
            self.rate[t] = p if t not in self.rate else (1 - self.alpha) * self.rate[t] + self.alpha * p
            self.seen[t] += len(xs)

    def weight(self, template):
        p = self.rate.get(template, self.prior)
        if self.is_mastered(template):
            return self.replay
        return self.floor + p * (1 - p) + (self.hard_bonus if p < 0.1 else 0.0)

    def is_mastered(self, template):
        return self.rate.get(template, 0.0) >= self.mastered and self.seen[template] >= self.mastered_seen

    def replay_weights(self):
        """Uniform over mastered trainable templates, or None if there are none yet."""
        names = [t for t in env.TEMPLATES if t not in env.HELDOUT_TEMPLATES and self.is_mastered(t)]
        return {t: (1.0 if t in names else 0.0) for t in env.TEMPLATES if t not in env.HELDOUT_TEMPLATES} if names else None

    def weights(self):
        return {t: self.weight(t) for t in env.TEMPLATES if t not in env.HELDOUT_TEMPLATES}

    def skill_rates(self, min_seen=1):
        """Solve rate per skill: templates sharing a skill, weighted by how often each was seen."""
        num, den = defaultdict(float), defaultdict(int)
        for t, p in self.rate.items():
            if self.seen[t] < min_seen:
                continue
            for s in env.SKILLS[t]:
                num[s] += p * self.seen[t]
                den[s] += self.seen[t]
        return {s: num[s] / den[s] for s in den}

    def weakest_skills(self, k, min_seen=16, below=0.6):
        """Skills worth generating new templates for: enough evidence, and not yet learned."""
        rates = self.skill_rates(min_seen)
        return sorted((s for s, p in rates.items() if p < below), key=rates.get)[:k]

    def evolve_targets(self, k, min_seen=16, below=0.6):
        """Weak skills first; when none is weak, the skills with the fewest trainable templates,
        then the highest solve rate, so generation continues after the pool is learned."""
        weak = self.weakest_skills(k, min_seen, below)
        rates = self.skill_rates(min_seen)
        count = {s: sum(s in env.SKILLS[t] for t in env.TEMPLATES if t not in env.HELDOUT_TEMPLATES)
                 for s in env.SKILL_DESCRIPTIONS}
        rest = sorted((s for s in count if s not in weak), key=lambda s: (count[s], -rates.get(s, 0.0), s))
        return (weak + rest)[:k]

    def snapshot(self):
        weights = self.weights()
        total = sum(weights.values())
        return {
            "templates": {t: {
                "rate": self.rate.get(t), "seen": self.seen[t], "share": weights[t] / total,
                "skills": list(env.SKILLS[t]), "tier": env.TEMPLATES[t][0], "generated": t not in env.BUILTIN,
            } for t in weights},
            "skills": self.skill_rates(),
        }
