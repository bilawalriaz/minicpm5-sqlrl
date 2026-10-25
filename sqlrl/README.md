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

The teacher uses the same tools, schema visibility, and turn budget. Verified
teacher actions are converted to the student's token format. Actions longer than
the student's token limit are excluded. Teacher injection supplies a supervised
positive gradient; it is not an on-policy sample from the student.

## Verifier and data

The current verifier is `sql-v2.2`; the answer protocol is `submit-sql-v2`. Each
submission must match the reference on four databases. Training and evaluation
use separate groups of four databases under `data/sql-v2.2/`:

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
their full precision. Queries have an instruction budget and a 10,000-row result
limit. Overflow returns an error; results are never silently truncated. A broken
reference query raises an error instead of grading the model. Graded SQL may not
use `random()`, `CURRENT_TIMESTAMP` and similar functions, or the `'now'`,
`'localtime'` and `'utc'` date modifiers.

## Evaluation

Evaluation uses the evaluation databases and reports three splits:

- `heldout_template`: `signups_year`, `orders_by_country` and `best_category_year`
  are never trained. Their skills appear in other templates. Each gets twice
  `--eval-per-template` questions.
- `reserved_params`: built-in parameter combinations excluded from training.
- `shared_wording`: the two constant templates. The wording is also trained; only
  the data is new.

`pass1` is attempt-weighted over all three. `by_split`, `by_template` and `by_tier`
count correct submissions, independently of reward shaping. Full evaluation
transcripts are saved in `runs/<name>/eval_samples/`. A checkpoint that descends
from a run trained on every built-in template (such as `curriculum_50b`) has seen
all of these questions, so its scores do not measure generalization.

## Generated templates

`evolve.py` proposes templates for weak skills. A template is accepted only when,
for every parameter combination (at most 256), the reference runs on every
fixture, its result does not depend on how `ORDER BY ... LIMIT` breaks ties, the
primary answer is not empty or NULL, the answers differ between fixtures, and the
results differ from every evaluation question. Removing the `HAVING` clause or any
top-level `WHERE` condition must change some answer. A solver that does not see the
reference must answer distinct questions. Held-out templates are not shown to the
generator. `python evolve.py --revalidate` applies the current data checks to
templates accepted earlier.

Previous databases and checkpoints are preserved. Template and curated-trace
stores live inside the versioned data directory; earlier approvals are not loaded
into a new verifier automatically. Run directories cannot be reused.

## Training settings

A training step samples 16 questions with 8 attempts each. The default learning
rate is 3e-5, LoRA rank 32, and LoRA scale 2.0. Temperature is 1.0, with five turns
and 256 generated tokens per turn. Gradient clipping is 1.0. A nonfinite loss
or gradient skips the update.

The M3 Pro profile uses `--gen-batch 64 --micro-tokens 8192 --mem-gb 26`.
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

Warm-start adapter weights with a new optimizer and a new run directory:

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

`--max-hours` stops before a step that would end after the budget, leaving time for
the scheduled and final evaluations. A nonfinite loss or gradient skips the update;
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

The watchdog sends SIGTERM after three checks in a row over 30 GiB footprint, over
3 GiB swap growth, or at critical memory pressure. The monitor writes
`runs/<name>/monitor.jsonl` every minute and sends SIGTERM if a literal answer is
graded correct, if evaluation falls 0.15 below both the first and best results
twice in a row, or if the status file stops changing for 45 minutes.

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
