# MiniCPM5 SQL agents

This repository trains OpenBMB's [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) and
[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) to answer questions about a SQLite database
by writing SQL, running it and submitting a final query. Training runs on one M3 Pro MacBook with
36 GB of memory: MLX, LoRA, GRPO, distillation from verified teacher solutions, and a curriculum that
generates new question types when the model runs out of things to learn.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/results-dark.svg">
  <img alt="Pass rates for released and fine-tuned MiniCPM5-1B and 2B on the test, check and selection sets and on trained types" src="docs/results-light.svg">
</picture>

Both models went through the same pipeline with the same data, seed and six-hour RL budget. Each
shop question was attempted 16 times at temperature 1, on databases the models never trained on.
Spider dev has 1,032 questions over 20 databases with other schemas, attempted four times each and
graded with Spider's test-suite rule. The baselines are OpenBMB's released models.

| | Released 1B | Fine-tuned 1B | Released 2B | Fine-tuned 2B |
| --- | --- | --- | --- | --- |
| Test set: 9 never-trained types, scored once | 3.0% | **47.9%** | 38.3% | **73.1%** |
| Check set: 5 never-trained types | 10.1% | **74.0%** | 73.6% | **96.2%** |
| Selection set: 5 never-trained types | 8.9% | **58.2%** | 47.4% | **76.0%** |
| Trained types, new values and data | 12.5% | **94.1%** | 69.5% | **98.8%** |
| [Spider dev](https://yale-lily.github.io/spider): 20 real schemas, never trained on | 6.6% | **29.2%** | 41.6% | **54.1%** |

The fine-tuned 1B passes the released 2B, a model twice its size, on every shop split except the
check set, where the two are level. Training used only the shop schema, and both models still
improved on every one of the 20 Spider databases.

Read the [blog post](https://bilawal.net/blog/minicpm5-1b-sql-agent/) for the full story with
charts, or [WRITEUP.md](WRITEUP.md) for the same results in this repository.

## What is in it

- **A grader that can't be gamed.** Every submission must match the reference on four databases,
  one of them built to catch plausible wrong SQL: duplicate names, prices on the thresholds,
  cancelled orders that change the winner. All 21 wrong rewrites in the tests are rejected, and
  correct rewrites pass. Regrading the first training run rejected 376 answers it had been rewarded for.
- **Evaluation that measures generalisation.** Separate evaluation databases, held-out question
  types for checkpoint selection and for checking, and a test set written after every design
  decision, scored once per model, with its exclusion rule fixed before any result existed.
- **Fixes for six ways RL made the model worse:** an exhausted training pool, forgetting,
  optimiser restarts, rewarded wrong answers, off-policy teacher examples, and policy swings.
  Each has a fix in `sqlrl/train.py` and a record in `sqlrl/engineering/verification.md`.
- **Generated training data.** Claude Haiku 5.5 and GLM-5.3 propose question templates, a separate
  model must solve them, and data checks reject the ones that would leak free reward. 252 accepted,
  552 rejected so far.
- **One pipeline for any model.** `sqlrl/scripts/pipeline.sh` scores the released model, distils,
  runs GRPO with early stopping, distils again, keeps the better checkpoint and writes a report.
- **Local deployment.** GGUF export for LM Studio and llama.cpp, with a system prompt and ten test
  prompts with measured success rates ([docs/lmstudio.md](docs/lmstudio.md)).

API spend for the whole project was about $2. Everything else ran on the laptop.

## Layout

| Path | Contents |
| --- | --- |
| `sqlrl/env.py` | Databases, question templates, the verifier and evaluation splits |
| `sqlrl/rollout.py` | Batched multi-turn rollouts with the `run_sql` and `submit_sql` tools |
| `sqlrl/train.py` | GRPO trainer: curriculum, teacher filter, EMA weights, early stopping |
| `sqlrl/sft.py` | Distillation on verified solutions with replay of the model's own answers |
| `sqlrl/evolve.py` | Template generation, solving, review and data checks |
| `sqlrl/evaluate.py`, `sqlrl/spider_eval.py` | Shop evaluation and Spider dev cross-schema evaluation |
| `sqlrl/scripts/` | Pipeline, monitoring, reports, charts and GGUF export |
| `sqlrl/tests/` | Verifier, training, rollout, curriculum and generation tests |
| `sqlrl/dashboard/` | Local dashboard for training metrics and rollout transcripts |
| `sqlrl/engineering/` | Engineering records and per-template reports for every reported number |

Setup, the answer protocol and every command are in [sqlrl/README.md](sqlrl/README.md). Run the
tests with `python -m pytest sqlrl/tests`.
