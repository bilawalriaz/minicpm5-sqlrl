# Verification record, 25 October 2026

Runtime: Apple M3 Pro, 36 GiB unified memory, 12 CPU cores, Python 3.13,
MLX 0.31.1 and mlx-lm 0.31.1. Dependency versions are pinned at the repository root.

The old curriculum_50b process was stopped after step 27. Its step-25 checkpoint,
raw run files, and original databases remain local. Training and evaluation data
for the new verifier are stored separately.

## Confirmed changes

- Explicit terminal submit_sql, complete answer storage, strict single-call parsing.
- Equal -0.2 rewards for token limits, turn limits, and missing submission.
- Full result comparison with explicit 10,000-row overflow errors.
- Positional columns, duplicate-sensitive rows, exact integers, no text stripping.
- Reference errors stop grading; known nondeterministic SQL functions are rejected.
- Four fixtures per split, independent historical sale prices, repeated customer
  names, sparse orders, and complete product-name coverage in one witness fixture.
- Explicit reserved parameters and separate evaluation databases. Constant-template
  wording is shared across splits. Evaluation does not duplicate question pairs.
- Separate versioned template and curated-trace stores; legacy approvals stay local.
- Student/teacher protocol alignment, schema visibility alignment, per-turn teacher
  token checks, recorded verifier reasons, and saved evaluation transcripts.
- Binary outcomes used for every evaluation accuracy, independent of reward shaping.
- Gradient finiteness checks, safe run-name reuse refusal, and adapter metadata preservation.

## Regression and integration checks

See pytest.txt, dashboard-check.txt, and dashboard-build.txt for executed checks.
The tests cover verifier failures, correct SQL alternatives, empty results, duplicate
names, complete product witnesses, resource budgets, protocol recovery and final-turn
submission, teacher action masks, evaluation calculations, and gradient accumulation
across different padding and micro-batch layouts.

A two-step integration run, verifier_v2_smoke, used the older sql-v2 fixture set
and the new submit-sql-v2 protocol. It warm-started curriculum_50b/step_00025 with
a fresh optimizer. Student training correctness was 3/16 then 7/16; the teacher
injected two then one verified trajectories. Fifteen-attempt evaluation accuracy
was 4/15, 11/15, then 13/15. These are pipeline observations on a small sample.
The run saved step checkpoints and final adapters and completed without nonfinite
updates. The per-tier evaluation values in that run were found to average shaped
rewards; the source was corrected and a regression test added. Raw artifacts are
preserved rather than rewritten.

## Archived SQL audit

legacy-sql-audit.json records 27 steps from curriculum_50b. The audit applies the
original three databases plus the four current training fixtures, preserving
previous counterexamples. Of 2,266 previously accepted SQL traces, 352 are rejected
by the stronger suite. All 471 previously rejected recorded submissions remain
rejected. Another 719 traces have no recorded final SQL and cannot be semantically
regraded. The audit includes student and teacher traces and evaluates SQL semantics,
not the new termination protocol.

## Hardware measurements

See hardware-benchmark.json and hardware-benchmark-32768.json. Each rollout case
uses 128 episodes and the same model checkpoint. Sampling consumption differs by
batch layout; compare token throughput as well as elapsed time. Gradient cases use
learning rate zero and do not change saved weights.

Generation batch 16: 261 tokens/s, 4.56 GiB MLX peak.
Generation batch 32: 291 tokens/s, 4.60 GiB MLX peak.
Generation batch 64: 497 tokens/s, 4.86 GiB MLX peak.
Gradient micro-batch 8,192: 113.8 s, 5.88 GiB MLX peak.
Gradient micro-batch 16,384: 115.3 s, 9.56 GiB MLX peak.
Gradient micro-batch 32,768: 120.8 s, 15.60 GiB MLX peak.
The 32,768-token measurement used a separately generated batch (6,875 tokens);
its throughput was 56.9 training tokens/s versus 61.6 for 8,192. Host swap usage
remained 1,224.12 MiB before and after that benchmark.

Generation batch 64 is selected. Micro-batch 8,192 remains the reference choice:
Neither larger micro-batch improved measured throughput. The MLX cap is 26 GiB. Learning rate
3e-5, rank 32, scale 2, temperature 1, 16 prompts, and group size 8 are retained;
this work does not establish an optimal learning rate from a hyperparameter sweep.

## Final integration run

verifier_v21_check completed two steps on the final sql-v2.1 verifier and
submit-sql-v2 protocol. It warm-started verifier_v2_smoke/final with a fresh
optimizer, generation batch 64, micro-batch 8,192, 26 GiB cap, four prompts and
eight samples per prompt. Evaluation used 22 distinct questions with two attempts
each: 39/44 before training, 34/44 after step 1, and 39/44 after step 2.
Step 1 had 32/32 correct and no nonzero-advantage groups, so no gradient update
occurred. Step 2 had 28/32 correct, 3/4 signal groups, gradient norm 0.211, and
5.70 GiB peak MLX memory. Step checkpoints and final adapters were saved.
The first run exercised actual teacher injection; this final run needed no rescue.
The stochastic evaluations are pipeline checks, not a learning-rate comparison.
Final-smoke.json stores configuration, metrics, and evaluation evidence.

The dashboard was rebuilt, its server restarted, and the run page returned HTTP
200. No overnight training has been started. The private GitHub repository contains
source, tests and these records; credentials, weights, databases and raw runs stay
local.

# Verification record, sql-v2.2, 25 October 2026

Same runtime as above. Previous data directories (`sql-v2`, `sql-v2.1`), runs and
checkpoints are unchanged. New fixtures are in `data/sql-v2.2/`.

## Confirmed bugs in sql-v2.1 and fixes

1. Accidental equivalences. `wrong-sql-v21-vs-v22.json` grades 21 plausible wrong
   rewrites of built-in references for every parameter on both splits. Eleven passed
   all four sql-v2.1 fixtures for some parameter: `COUNT(DISTINCT name)` for a city
   count (15/16 train, 16/16 eval), `<=` for `<` (18/18), an open-ended year range
   for 2024 signups (5/15), `AVG(DISTINCT price)`, `SELECT DISTINCT name` in
   frequent_customers (4/4) and top_spenders (2/2), `GROUP BY name` in top_spenders
   (eval 2/2), a missing status filter and `SUM(quantity)` in best_category_year
   (train 2/3 and 1/3), and missing `COALESCE` or `SUM(DISTINCT ...)` in
   units_of_product (eval 3/198 and 1/198). The sql-v2.2 adversarial fixture adds
   the cases listed in README.md; all 21 rewrites are rejected for every parameter.
   Equivalent rewrites (`NOT EXISTS`, `LEFT JOIN ... IS NULL`, `BETWEEN`, `TOTAL`,
   `IN` subqueries, reordered expressions) are accepted.
2. Clock reads. `date('now')`, `strftime('%s','now')` and `julianday('now')` passed the
   deterministic check because the authorizer sees only function names. Graded SQL
   with `'now'`, `'localtime'` or `'utc'` is now rejected.
3. Generated-template validation accepted references whose answers were empty,
   identical on every fixture (a copied value passes), tie-dependent, or equal to an
   evaluation question in other words. Two legacy generated templates from
   curriculum_50b have the same answer on all four sql-v2.2 fixtures; the new check
   rejects them. A clause-ablation check rejects templates whose `HAVING` or `WHERE`
   condition never changes an answer; it rejected `big_revenue_categories`
   (thresholds 1,000 to 10,000 against category totals in the hundreds of thousands).
4. Training metrics `by_template` and `by_tier` averaged shaped rewards, and the
   dashboard showed them as rates beside evaluation solve rates. They now count
   correct submissions. The dashboard labels older runs "Reward".
5. A nonfinite update raised an exception and ended the run. It now skips the update;
   three consecutive skips stop the run.
6. Checkpoints were written in place and held no optimizer state. They are now
   written to a temporary directory and renamed, and include optimizer and
   curriculum state. `--resume` restores them. SIGTERM or SIGINT used to leave a
   stale status file and no checkpoint; the trainer now finishes an in-progress
   update or save, saves `interrupted_step_NNNNN/`, and records the error.

## Checked and found consistent

- KV-cache continuation. mlx-lm 0.31.1 `BatchGenerator` runs each sampled token
  through the model before it reports it, so a finished sequence's cache includes its
  stop token. `BatchKVCache` keeps per-sequence RoPE offsets. `OBS_TEMPLATE` matches
  the MiniCPM5 chat template for tool messages and the no-think generation prompt.
- Sampling and training probabilities. `logprob-check.json`: 48 episodes from
  curriculum_50b/step_00025, 6,646 generated tokens, all multi-turn. Mean absolute
  difference between sampler and training-forward log-probabilities is 0.0087 nats
  (p99 0.136, max 0.74); later-turn tokens 0.0085. Dropping each first-turn stop token
  raises later-turn disagreement to 0.0154, so this control is weak evidence; a
  position shift would be larger.
- Target shift, masks, right padding and micro-batch accumulation (existing tests).
- Teacher actions use the student's exact call format and token budget (inspected).
- No built-in reference depends on tie order (random secondary sort keys, 12 tries
  per fixture), and no built-in answer is identical on all four fixtures (tests).

## Evaluation split

Held-out templates: `signups_year`, `orders_by_country`, `best_category_year`.
`checkpoint-eval-v22.txt` evaluates existing adapters (29 questions, 4 attempts):

| Adapter | pass@1 | held-out templates | reserved params | shared wording |
| --- | --- | --- | --- | --- |
| Base model | 0.078 | 0.068 | 0.078 | 0.125 |
| curriculum_50b/step_00025 | 0.319 | 0.386 | 0.250 | 0.500 |
| verifier_v21_check/final | 0.931 | 0.977 | 0.906 | 0.875 |

The curriculum_50b lineage trained on every built-in template and parameter, so its
scores do not measure generalization. The overnight run starts from the base model
with a fresh LoRA adapter.

## Archive regrading

`sql-audit-curriculum_50b.json` grades 27 steps on the three original databases plus
each version's training fixtures. Of 2,266 recorded accepts, 1,886 pass both
versions, 348 fail both, 28 pass sql-v2.1 but fail sql-v2.2 (frequent_customers 15,
top_spenders 9, legacy generated templates 3, priciest_in_cat 1), and 4 fail sql-v2.1
but pass sql-v2.2. Those 4 belong to the two legacy generated templates above; the
new template validation rejects both templates. All 471 recorded rejects stay
rejected; 719 traces have no recorded SQL. Teacher traces (55) pass both versions.
`sql-audit-v2-runs.json`: the 84 recorded submissions in verifier_v2_smoke and
verifier_v21_check keep their verdicts.

## Short runs on the final code

`short-runs-v22.json` holds configuration, per-step metrics and evaluations.

- v22_base_short: base model, 16 prompts x 8, teacher on, evolution off, 10 steps.
  Evaluation (29 questions x 2): 0.086, 0.345 at step 5, 0.707 at step 10
  (held-out templates 0.091, 0.364, 0.545). Training correctness 33/128 to 108/128.
  Groups with mixed rewards: 16, 15, 16, 11, 14, 14, 11, 11, 10, 10. The teacher
  rescued 5 and 9 groups in steps 1 and 2. Missing submissions fell from 57 to 17
  after one update and to 0 by step 4. Mean turns fell from 3.2 to 1.0: the model
  submits at once rather than risk the -0.2 turn-limit penalty. Step time fell from
  about 210 s to 80 s as episodes became single-turn. Peak MLX memory 9.2 GiB. No
  literal answer was graded correct. Inspected rejected answers were wrong
  (missing filters, wrong joins, correlated-subquery mistakes, syntax errors).
- v22_ctl_time: `--max-hours 0.08`; stopped after step 8 at 0.078 h with
  `stop_reason: time limit`, ran the final evaluation and saved `final`.
- v22_ctl_resume: `--resume v22_ctl_time/final`; config records
  `optimizer_restored: true`, `start_step: 8`. SIGTERM during the step-10 rollout
  saved `interrupted_step_00009` and set the status to error.

Template evolution ran once against the API: 4 of 6 proposals accepted; one later
removed by the clause-ablation check. Three remain accepted for training.

## Checks

`pytest.txt` (39 passed), `dashboard-check.txt`, `dashboard-build.txt`.
