# Project instructions

Use simple technical English. State what a feature does and how to use it.
Do not add unsolicited defensive prose, reassurance, or liability-style caveats.
Keep practical requirements and troubleshooting concise. Keep validation evidence
in engineering records. Never invent successful validation.
Use ~/.codex/skills/humanizer/SKILL.md when writing public copy, if available.

Training code is in sqlrl/. Run Python tests with `python -m pytest sqlrl/tests`.
For dashboard changes, read sqlrl/dashboard/AGENTS.md and run `pnpm check` and
`pnpm build` in that directory.

Keep credentials, local model weights, generated databases, and run artifacts out
of Git. Preserve previous checkpoints and use a new directory for every run.
Do not run two GPU training or benchmark processes concurrently.
