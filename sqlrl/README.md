# SQL agent training

The agent explores a shop database with `run_sql`, then answers with `submit_sql`.
The complete result of the submitted query is the answer. Tool previews show ten
rows; they do not limit the answer. Plain text does not submit an answer.

## Setup

Use Apple Silicon and Python 3.13. From the repository root:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c 'from huggingface_hub import snapshot_download; snapshot_download("openbmb/MiniCPM5-1B", local_dir="models/MiniCPM5-1B")'
cd sqlrl
python env.py
cd dashboard
pnpm install --frozen-lockfile
pnpm build
cd ..
python serve.py
```

Open http://localhost:8000. Run training in another terminal with the same Python
environment:

```sh
cd sqlrl
python train.py --name first_v2 --steps 50
```

For teacher rescue and generated templates, copy `.env.example` to `.env` in
`sqlrl/` and set `OPENCODE_GO_API_KEY`. The key is read from `.env` before the
shell environment. Enable the teacher with `--teacher glm-5.3-flash`. Evolution
runs every ten steps; use `--evolve-every 0` to disable it.

## Answer and reward

- `run_sql` explores or checks a query. It does not submit an answer.
- `submit_sql` submits exactly the requested columns in the requested order and
  ends the episode. Submission works on the final allowed turn.
- A correct submission earns 1, minus the configured prose-token penalty, capped
  at 0.3. An incorrect submission earns 0.
- Plain-text termination, token-limit termination, and exhausting the turn budget
  without submission all earn `--no-answer-reward` (default -0.2).
- Malformed, multiple, or unknown function calls receive a tool error. The model
  can recover on a later turn.

Advantages are rewards minus the group mean, with two limits. In a group where no
student answered correctly, students get no gradient. In other groups, an answer
that is not correct never gets a positive advantage. Without these limits, wrong SQL
outranked unanswered attempts and was reinforced.

The teacher uses the same tools, schema visibility, and turn budget. Verified
teacher actions are converted to the student's token format. Actions longer than
the student's token limit are excluded. Teacher injection supplies a supervised
positive gradient; it is not an on-policy sample from the student. It helped the
base model learn the protocol, but with a large bank of hard templates it degraded
skills the model already had, so later runs leave `--teacher` off (see
`engineering/verification.md`).

## Verifier and data

The current verifier is `sql-v2.3`; the answer protocol is `submit-sql-v2`. Each
submission must match the reference on four databases. Training and evaluation
use separate groups of four databases under `data/sql-v2.2/` (`sql-v2.3` changes
only the instruction budget and keeps those fixtures):

| Role | Contents |
| --- | --- |
| Primary | Random catalog and orders. Historical sale prices differ from catalog prices. |
| Adversarial | Repeated customer names in every city, two same-name customers who both exceed every order and spend threshold, prices exactly at question thresholds, a repeated price per category, signups and orders outside the usual years, and cancelled or cheap sales that change the category ranking only for wrong SQL. Half of the products have no sales. |
| Sales witness | Every product sells, with repeated quantities. |
| Sparse | 80 orders; the other half of the products have no sales. |

Every built-in product name has zero sales on some fixture and non-zero sales on
another. `tests/test_verifier.py` lists 21 plausible wrong rewrites of the built-in
references (for example `COUNT(DISTINCT name)`, `<=` for `<`, `GROUP BY name`,
`SELECT DISTINCT name`, a missing status filter). Eleven of them passed every
`sql-v2.1` fixture for at least one parameter; the rest guard earlier fixes. All
are rejected for every parameter on both splits. Equivalent rewrites (`NOT EXISTS`, `LEFT JOIN ... IS NULL`, `BETWEEN`,
`TOTAL`, `IN` subqueries) are accepted.

Comparison preserves column positions, duplicate rows, and text values. Row order
is ignored. Numeric results are compared to six decimal places. Integers retain
their full precision. Queries have a budget of 50 million SQLite instructions
(about 18 million for a correct correlated subquery over unindexed order lines,
under 0.1 million for the references) and a 10,000-row result limit. Overflow returns an error; results are never silently truncated. A broken
reference query raises an error instead of grading the model. Graded SQL may not
use `random()`, `CURRENT_TIMESTAMP` and similar functions, or the `'now'`,
`'localtime'` and `'utc'` date modifiers.

## Evaluation

Evaluation (`eval-v3`) uses the evaluation databases. Ten built-in templates are
never trained; their skills appear in other templates. They form two sets:

- Selection set (`heldout_select`): `signups_year`, `orders_by_country`,
  `best_category_year`, `lines_per_order_status`, `repeat_buyers_cat`. Scored at
  every evaluation; it chooses the checkpoint and drives early stopping.
- Check set (`heldout_check`): `country_revenue_year`, `idle_customers_country`,
  `status_counts_year`, `large_orders`, `first_order_city`. Scored before training
  and on the final and selected checkpoints only, so selection does not bias it.

Each held-out template contributes `--eval-heldout-questions` questions (default 4)
with `--eval-heldout-k` attempts (default 8). Every evaluation also scores trained
templates: `reserved_params` (parameter combinations excluded from training,
`--eval-per-template` questions, `--eval-k` attempts) and `shared_wording` (the two
constant templates; only the data is new).

Results go to `evals.jsonl` and check results to `checks.jsonl`. `by_split`,
`by_template` and `by_tier` count correct submissions, independently of reward
shaping. Full transcripts are saved in `runs/<name>/eval_samples/`. A checkpoint that
descends from a run trained on every built-in template (such as `curriculum_50b`) has
seen these questions, so its scores do not measure generalization.

## Cross-schema evaluation

`spider_eval.py` measures transfer to schemas the model never trained on. It runs
Spider dev questions through the same agent loop, on the original Spider database
with its schema in the system prompt. Each submission is graded on that database and
all of its Spider test-suite variants, by the strict verifier rule and by Spider's
test-suite rule (columns may be permuted; row order counts when the reference has
ORDER BY).

```sh
python scripts/prepare_spider.py   # downloads about 1.3 GB, extracts 3.3 GB
python spider_eval.py --name spider_1b_released
python spider_eval.py --name spider_2b_released --model ../models/MiniCPM5-2B
```

Results go to `runs/<name>/` (`summary.json`, `episodes.jsonl`, `excluded.json`).
Use `--db` or `--limit` for a smaller run and `--check-only` to test the reference
queries without a model.

## Generated templates

`evolve.py` proposes templates for weak skills. A template is accepted only when,
for every parameter combination (at most 256), the reference runs on every
fixture, its result does not depend on how `ORDER BY ... LIMIT` breaks ties, the
primary answer is not empty or NULL, the answers differ between fixtures, and the
results differ from every evaluation question. Removing the `HAVING` clause or any
top-level `WHERE` condition must change some answer. A solver that does not see the
reference must answer distinct questions. Held-out templates are not shown to the
generator. A generated question is rejected when, on every training fixture, its
answer equals a held-out answer, is contained in one (one status of a per-status
count), or contains a held-out answer's non-integer numbers or dates. Templates that
reuse a held-out template's skills with other columns or filters are allowed. `python evolve.py --revalidate` applies the current data checks to
templates accepted earlier.

Build a template bank before a run, then let training add more:

```sh
python evolve.py --skills all --n 4 --rounds 3
```

During training, generation runs every `--evolve-every` steps for
`--evolve-skills` skills: skills solved less than 60% of the time first, otherwise
the skills with the fewest trainable templates. Proposers rotate through
`--evolve-model` (Claude Haiku 5.5 through the gateway's Messages endpoint, GLM-5.3,
GLM-5.3-flash) and `--evolve-solver` must solve each proposal without the
reference. When the mean number of groups with signal over `--signal-window` steps
falls below `--signal-floor`, a burst generates templates for `--burst-skills`
skills: harder compositions when all-correct groups outnumber all-fail groups,
ordinary ones otherwise.

A template solved at least 95% of the time over 32 rollouts drops from the sampling
floor (0.05) to a weight of 0.01. Mastered templates instead fill `--replay-share`
of each batch (default 0.25), drawn uniformly, so the model keeps them. A warm start
inherits the curriculum rates stored with the checkpoint.

Previous databases and checkpoints are preserved. Template and curated-trace
stores live inside the versioned data directory; earlier approvals are not loaded
into a new verifier automatically. Run directories cannot be reused.

## Training settings

A training step samples 16 questions with 8 attempts each. The default learning
rate is 3e-5, LoRA rank 32, and LoRA scale 2.0. Temperature is 1.0, with five turns
and 256 generated tokens per turn. Gradient clipping is 1.0. A nonfinite loss
or gradient skips the update. `--lr-end` decays the learning rate linearly to that
value at `--steps`. `sample_nll` in `metrics.jsonl` is the mean negative
log-probability of sampled tokens; a sustained rise means the policy is getting
noisier.

The M3 Pro profile uses `--gen-batch 64 --micro-tokens 8192 --mem-gb 30` (`--prompts 32 --gen-batch 128` for long runs).
`--gen-batch` controls parallel decoding. `--micro-tokens` controls padded tokens
per gradient micro-batch. `--mem-gb` caps MLX memory; the wired limit also respects
Apple's recommended working set. Measure batch sizes on the target machine with:

```sh
python benchmark.py --adapter runs/first_v2/final
```

The benchmark compares generation batches of 16, 32, and 64 and gradient
micro-batches of 8,192, 16,384, and 32,768 tokens. Gradient measurements use learning rate
zero and do not change saved weights. Results go to
`engineering/hardware-benchmark.json`.

Warm-start adapter weights with a new optimizer and a new run directory (a fresh
Adam state makes large early updates on a trained adapter; prefer `--resume`):

```sh
python train.py --name continued_v2 --adapter runs/first_v2/final --steps 50
```

Continue a checkpoint with its optimizer state, step count and curriculum:

```sh
python train.py --name first_v2_part2 --resume runs/first_v2/step_00050 --steps 100
```

Each checkpoint holds `adapters.safetensors`, `optimizer.safetensors`,
`adapter_config.json` and `trainer_state.json`. It is written to a temporary
directory and renamed when complete. Checkpoints are written every `--save-every`
steps (default 25) and at the end; evaluation runs before training, every
`--eval-every` steps, and after the last step.

Every evaluated step keeps a checkpoint. The trainer smooths the selection score
over two consecutive evaluations and stops early after `--patience` evaluations
without a better smoothed score (default 5), or when the score stays `--drop-stop`
below its best for `--drop-evals` evaluations (defaults 0.15 and 3). At the end it
writes `selected.json`, then scores the check set on the final and selected
checkpoints.

`--max-hours` stops before a step that would end after the budget, leaving time for
the scheduled evaluation, the final evaluation and the two checks. A nonfinite loss or gradient skips the update;
`--max-nonfinite` consecutive skips (default 3) stop the run. SIGTERM or SIGINT
during an update or a save takes effect after it completes. On any failure the
trainer saves `interrupted_step_NNNNN/` with the weights of the last completed
update and marks `status.json` as an error.

Run beside a long training run:

```sh
caffeinate -i -w <training-pid> &
scripts/watchdog.sh <training-pid> runs/<name>.mem &
python scripts/monitor.py --run <name> --pid <training-pid> &
```

The watchdog sends SIGTERM after three checks in a row over its footprint limit (pass 33792 MB with `--mem-gb 30`), over
3 GiB swap growth, or at critical memory pressure. The monitor writes
`runs/<name>/monitor.jsonl` every minute and sends SIGTERM if a literal answer is
graded correct, if evaluation falls 0.15 below both the first and best results
twice in a row, or if the status file stops changing for 45 minutes. It records an
alert when the selection score falls 0.15 below its best.

## Checks and inspection

```sh
python -m pytest tests -q
python evaluate.py --adapter runs/first_v2/final --out runs/first_v2/evaluation.jsonl
python audit_rollouts.py --run curriculum_50b --out engineering/sql-audit-curriculum_50b.json
python scripts/logprob_check.py --adapter runs/first_v2/final --out engineering/logprob-check.json
python curate.py ingest
python curate.py review --limit 64 --rejects 16
python curate.py report
cd dashboard
pnpm check
pnpm build
```

`audit_rollouts.py` grades archived SQL on the original databases plus the
`sql-v2.1` and current training fixtures. It measures query semantics, not
compatibility with the submission protocol. `scripts/logprob_check.py` compares the
sampler's token log-probabilities with a full-sequence training forward pass. `curate.py` imports only traces with the current verifier and protocol
versions. The `safe_traces` view contains traces approved by both checks.

The dashboard serves `/api/runs`, `/api/runs/<name>`, archived training samples,
and `/api/runs/<name>/live`. Old run scores retain their original meaning; inspect
the verifier and protocol versions when comparing runs.
