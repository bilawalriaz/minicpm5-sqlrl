# Plan: targeted continuation from the final checkpoints

Status: planned, not started. Written 26 October 2026 after the Spider dev runs.

The pipeline ends with a final scoring. This stage reads that scoring, picks the largest
weaknesses, builds training data for them, and continues training from the final
checkpoints. Each evaluation then defines the next round of training.

## Weaknesses to target

From `release-1b-pipeline-report.md`, `release-2b-report.md` and `spider-dev-report.md`:

1. Failed queries on unfamiliar schemas. On Spider dev, 1,491 of the 1B's 2,717 wrong
   answers do not execute (947 "no such column"). In 864 the last `run_sql` returned an
   error and the model submitted anyway; in 509 it ran nothing before submitting. The 2B
   has 146 of 1,770.
2. Weak shop question types: busiest month of a year (1B 0%, 2B 27%), share of returned
   orders (0%, 5%), lines per order by status (2%, 27%), and counting groups that pass a
   `HAVING` filter (1B 21-35%).

## Data

Multi-schema tasks from Spider train:

- Download the Spider train databases from the official release (`spider_data.zip`, linked
  from https://yale-lily.github.io/spider). None of the train schemas is one of the 20 dev
  schemas, so Spider dev stays an unseen-schema test.
- Give each train database three variants made by keeping a random 70% of the rows of each
  table, with foreign keys kept consistent. Grade on the original and all variants, so a
  hard-coded answer cannot pass.
- Keep a question only when its reference runs on every variant, returns a non-empty
  result, and gives different answers on at least two variants (the shop template rule).
- Show the Spider schema in the system prompt, as `spider_eval.py` does.

Teacher solutions:

- `scripts/collect_solutions.py` extended to Spider train tasks: GLM-5.3-flash solves each
  question in the same `run_sql`/`submit_sql` loop; keep solutions verified on every
  variant. Keep transcripts where the teacher hit an execution error and recovered.

Weak shop types:

- `evolve.py --skills date_filter,rounding,having` with stepping stones for month-of-year
  aggregation, ratios and counts over a `HAVING` filter. The held-out containment check
  still applies, so no generated template may reproduce a held-out answer.

## Training

For each model, from its final checkpoint (`runs/p1b_sft2/final`, `runs/p2b_sft2/final`):

1. Distillation with `--resume`: Spider train solutions plus the new shop solutions, with
   replay of the model's own correct shop transcripts.
2. GRPO for about two hours on a mix of shop tasks (with the 25% replay share) and Spider
   train tasks.
3. Keep the better of the two by the selection set, as in `pipeline.sh`.

## Evaluation

- Every shop split, to measure regressions on what the model already does.
- Spider dev, unchanged.
- The shop test set has been scored once per model. The continuation is scored on it once
  more for comparison, but it is never used to choose between continuation checkpoints.

## Code changes

- `spider_train.py` (new): download check, variant builder, question filter, task source.
- `train.py`: `--spider-share` for the share of Spider train tasks per batch.
- `scripts/collect_solutions.py`: Spider train task source.
- `scripts/pipeline.sh` or a new `scripts/continue.sh`: the stage above.
- Tests: variant consistency, question filter, grading across variants.

## Budget

About 8-9 hours of wall time for both models, one GPU process at a time. Expected API
spend under $1.
