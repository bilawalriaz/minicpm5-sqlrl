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

## Overnight run: overnight_v22_base

Started 04:54 on 25 October 2026 at commit f2df4bf (`git` field in config.json):

```sh
python -u train.py --name overnight_v22_base --steps 300 --max-hours 8 \
  --teacher glm-5.3-flash --eval-every 10 --eval-k 4 --save-every 10 --seed 11
```

Base model with a fresh LoRA adapter (rank 32, scale 2), learning rate 3e-5, 16
prompts x 8, temperature 1, five turns, 256 tokens per turn, gradient clipping 1,
generation batch 64, micro-batch 8,192 tokens, 26 GiB MLX cap, evolution every 10
steps. Short-run step times were 80 to 210 s and evaluation 11 to 52 s, so about
280 steps fit in eight hours; the wall-clock bound stops earlier if steps are
slower. Checkpoints with optimizer state go to `runs/overnight_v22_base/step_NNNNN/`
every 10 steps and to `final/` at the end.

Attached processes: `caffeinate -i -s -m -w`, `scripts/watchdog.sh` (log
`runs/overnight_v22_base.mem`) and `scripts/monitor.py` (log
`runs/overnight_v22_base/monitor.jsonl`). The first monitor process exited at
startup because the run directory did not exist yet
(`runs/overnight_v22_base.monitor-crash1.log`); it was fixed in 98ebdec and restarted
at step 4. The dashboard server from the earlier session had exited; it was
restarted and `/runs/overnight_v22_base` returned HTTP 200.

First observations: evaluation pass@1 0.129 at step 0 and 0.543 at step 10
(held-out templates 0.091 then 0.227; reserved parameters 0.141 then 0.703).
Training reward rose from -0.091 to 0.789. Missing submissions fell from 64 to 0 by
step 3. Peak MLX memory 7.3 GiB. The first evolution cycle accepted 1 of 4
templates (`avg_spend_by_city_on_category`); the rejections were an empty
primary answer (twice) and more than 256 parameter combinations.

### Outcome

The run completed all 300 steps in 5.33 hours (`stop_reason: completed`), faster
than the eight-hour bound because episodes became two-turn. Checkpoints
`step_00010` to `step_00300` and `final/` are saved. The monitor recorded no alerts
and no stop; the watchdog never hit a limit; no update was skipped; no literal answer
was graded correct. Peak MLX memory was 9.7 GiB. Teacher cost was $0.059 and
evolution $0.007; evolution accepted 7 templates in 30 cycles.

| Steps | Mean reward | Groups with signal (of 16) | Teacher rescues | Seconds per step |
| --- | --- | --- | --- | --- |
| 1-50 | 0.679 | 9.8 | 90 | 110 |
| 51-100 | 0.956 | 3.1 | 3 | 57 |
| 101-200 | 0.931 | 3.5 | 19 | 60 |
| 201-300 | 0.991 | 1.1 | 0 | 40 |

Evaluation (29 questions x 4): pass@1 0.129 at step 0, 0.810 at step 40, highest
0.871 at step 160, 0.767 at step 300. Reserved parameters reached 0.94 by step 30
and 1.00 at most later evaluations. Held-out templates rose to 0.682 at step 160,
then varied between 0.045 (step 240) and 0.523; 0.386 at step 300. By template at
steps 160 / 240 / 300: signups_year 15/16, 0/16, 1/16; orders_by_country 15/16,
2/16, 16/16; best_category_year 0/12 at all three. Late signups_year failures join
orders or filter on order dates, the pattern of trained templates.

Interpretation: the training pool was saturated by about step 60. After that, few
groups carried gradient and in-distribution scores stayed near 1.0 while transfer
to held-out templates fell and became unstable. Each held-out estimate uses 44
attempts on 11 questions, so single evaluations are noisy, but the signups_year
change (15/16 to 0-1/16) is large. The monitor's collapse rule uses overall pass@1,
which stayed within 0.15 of the first evaluation, so it did not stop the run.
`step_00160` has the best held-out score; it was selected on the same evaluation
set, so that score is optimistic.

# Follow-up: generalization, 25 October 2026 (eval-v3, sql-v2.3)

## Changes

- eval-v3: a selection set (5 never-trained templates; checkpoint choice and early
  stopping) and a check set (5 other never-trained templates; scored only at the
  start and on the final and selected checkpoints). Seven held-out templates are new.
- sql-v2.3 raises the instruction budget to 50M; a correct correlated subquery over
  unindexed order lines needs about 18M. 30 of 41,665 overnight submissions hit the
  old budget; those inspected were wrong for other reasons.
- Template bank: 124 validated generated templates before the long run. Proposers
  rotate Claude Haiku 5.5 (gateway Messages endpoint), GLM-5.3 and GLM-5.3-flash; a
  separate solver checks each proposal. A containment check rejects generated
  answers that are part of, or contain, a held-out answer (it rejected
  `count_orders_status_year`, one row of the held-out `status_counts_year`).

## Clean comparison on the check set

`check-set-overnight.txt` (18 questions x 8 attempts; never trained or used for
selection): base model 0.132, overnight step 40 0.146, step 160 0.403, final 0.389.

## Short validation runs (all resumed or warm-started from overnight step 160)

| Run | Change | Selection, start then later | Reserved params | Sampled-token NLL |
| --- | --- | --- | --- | --- |
| v23_short | base model, full bank | 0.053 -> 0.092 (step 4) | 0.06 -> 0.19 | n/a |
| v23_short_warm | fresh optimizer, 0.01 replay weight | 0.38 -> 0.22 (step 8) | 1.00 -> 0.58 | n/a |
| v23_replay_check | + 25% replay of mastered templates | 0.36 -> 0.33 (step 8) | 0.98 -> 0.56 | n/a |
| v23_resume_check | restored optimizer state | 0.37 -> 0.33 (step 172) | 0.98 -> 0.63 | 0.05 -> 0.55 |
| v23_fix_check | wrong answers never positive | 0.35 -> 0.22 (step 168) | 0.98 -> 0.67 | 0.03 -> 0.12 |
| v23_noteacher_check | teacher off | 0.35 -> 0.41 -> 0.18 (176) | 0.98 -> 0.94 | flat 0.03-0.06 |
| v23_ema_check | near-policy teacher x0.5, EMA 0.9, lr 1.5e-5 | 0.375 -> 0.421 (172), 0.408 (180) | 0.98 -> 0.98-1.00 | flat 0.04-0.08 |

Findings, in order: mastered templates need a fixed replay share; a fresh Adam state
on a trained adapter makes large noisy updates; in groups with no correct answer,
wrong SQL ranked above unanswered attempts was reinforced (3-21 per step); teacher
solutions far from the student's distribution corrupted existing skills (invented
tables and joins); small batches over a varied pool make the policy switch SQL
styles between steps (held-out 0.41 -> 0.18 in 4 steps without averaging).

v23_ema_check check set: 0.396 at step 160, 0.590 final (step 180), 0.569 on the
selected step 176. Teacher solutions offered had 0.23-1.7 NLL per token; the 1.0
filter kept about 80%.

## Long run

final_v23 resumes v23_ema_check/final with 32 prompts x 8, generation batch 128,
lr 1.5e-5 decaying to 5e-6, evaluation every 10 steps, patience 6, 8-hour bound,
near-policy teacher, EMA 0.9, replay 0.25, stepping stones and bursts.

## Test-set protocol (written before any test-set result)

Ten templates (`env.HELDOUT_TEST`) were written during final_v23b, after the
development decisions. final_v23b loaded its templates at launch, so it cannot sample
them. They are scored once, after training, on the base model and on the
checkpoint chosen by the selection set, with
`python evaluate.py --sets test --k 16 --heldout-questions 4`.

Exclusion rule, fixed now: after training, every accepted generated template is
compared with each test question using `evolve.contained` on the training fixtures.
A test template with any overlapping question is excluded from the headline number
and listed separately. No other test template is dropped, and the test set is not
used to choose a checkpoint or setting.

## Distillation on the 1B model

sft_1b_v1 resumed final_v23b/step_00200 (best smoothed selection score in that run;
final_v23b was stopped after three flat evaluations: 0.454, 0.447, 0.447). Data: 787
verified GLM-5.3-flash solutions over 212 trainable templates plus 567 correct
student transcripts from final_v23b and v23_ema_check; all passed the 1.5 NLL filter
(median 0.078). 86 updates at lr 1e-5, two epochs, EMA 0.9.

| | Before | After |
| --- | --- | --- |
| Selection set | 0.461 | 0.513 |
| Check set | 0.653 | 0.722 |
| Reserved parameters | 0.969 | 0.969 |

sft_1b_v1/final is the final 1B checkpoint (higher selection score).

## Test-set exclusion

`test-set-exclusions.txt` applies the recorded rule to 202 accepted generated
templates: `avg_order_value_country` overlaps `avg_order_value_by_country_status`
and is excluded from headline test numbers. The other nine test templates are kept.

## Released 2B baseline

Stage 1 of `scripts/pipeline.sh ../models/MiniCPM5-2B p2b 6` scored the released
MiniCPM5-2B (bf16) on every split with k 16 and 384 tokens per turn
(`runs/p2b_results/released.log`). `released-2b-report.md` compares it with the
released 1B, with `avg_order_value_country` excluded from the test set.

| Split | Released 2B | Released 1B |
| --- | --- | --- |
| Test set | 0.383 | 0.036 |
| Check set | 0.736 | 0.122 |
| Selection set | 0.474 | 0.056 |
| Reserved parameters | 0.695 | 0.148 |

2B outcomes over 1,472 attempts: 773 correct, 459 wrong, 192 max turns, 48 no query
(1B: 118 correct, 100 wrong, 507 max turns, 640 no query, 107 length).

## Cross-schema evaluation: Spider dev (spider-dev-testsuite-v1)

Every evaluation split above uses the shop schema. `spider_eval.py` runs Spider dev
(1,034 questions, 20 databases) through the same `run_sql`/`submit_sql` rollout. The
agent sees the original Spider database and its schema; the system prompt drops "of an
online shop" and shows that schema. Each submission is graded on the original database
and every Spider test-suite variant (25 to 60 databases per schema including the original,
695 files), with two
rules: strict (`env.grade`, column positions kept) and test-suite (`exec_eval.result_eq`
with `--keep_distinct`: columns may be permuted, row order counts when the reference has
ORDER BY). `scripts/prepare_spider.py` downloads the data into `data/spider/`.

Reference check (`python spider_eval.py --check-only`): 1,032 references run on every
database; 2 are excluded because they return over 10,000 rows (`wta_1` #455 and #456).
Each usable reference, submitted as the answer, passes both rules on every database.
`tests/test_spider_eval.py` covers the comparison rule, exclusion, grading on variants,
and the unchanged shop prompt.

Runs (k 4, temperature 1, 5 turns, 384 tokens per turn), queued after the 2B pipeline:
`runs/spider_1b_released`, `runs/spider_2b_released`, `runs/spider_1b_sql`
(`release/MiniCPM5-1B-SQL`), 26 October 2026. 1,032 questions, 4,128 attempts each.
Intervals resample questions.

| | Released 1B | Released 2B | Fine-tuned 1B |
| --- | --- | --- | --- |
| Test-suite pass@1 | 6.6% (5.6-7.7%) | 41.6% (39.2-44.2%) | 25.9% (23.7-28.1%) |
| Strict pass@1 | 6.5% (5.5-7.5%) | 38.7% (36.2-41.1%) | 25.3% (23.1-27.5%) |
| Test-suite pass@4 | 17.2% | 58.8% | 40.7% |
| Correct / wrong | 267 / 316 | 1,596 / 1,586 | 1,043 / 3,026 |
| No query / max turns / length | 1,992 / 1,161 / 392 | 322 / 621 / 3 | 0 / 50 / 9 |
| Mean turns | 3.54 | 2.72 | 1.97 |
| Wrong submissions that are SQL errors | 142 | 67 | 2,096 |

Shop training transfers: the fine-tuned 1B is ahead of the released 1B on every
Spider database. It is below the released 2B on 18 of 20, level on
`real_estate_properties` (4 questions) and ahead on `network_1` (0.46 vs 0.43).

The fine-tuned 1B submits on almost every attempt, as on the shop, but 2,096 of its
3,026 wrong submissions fail to execute: 1,350 "no such column", 393 syntax errors,
207 "no such table". In 1,312 of the 2,096 its last `run_sql` returned an error and it
submitted a different query without testing it; in 370 it submitted the exact query
that had just failed; in 332 it ran nothing before submitting. The released 2B has 67
such errors. Typical case (`concert_singer`): it joins `concert` on a `Singer_ID`
column that table does not have, reads the error, and submits the same join. Likely
cause (not measured): the shop curriculum rarely produces an execution error, so
recovery from one was never trained.

## Real-question seeds for template generation

`scripts/fetch_seeds.py` downloads 1,500 question/SQL pairs from Spider's train split
(Hugging Face `xlangai/spider`). Spider dev stays the cross-schema evaluation set and
is never used for training. `evolve.py --seeds` shows four random pairs with each
proposal as examples of phrasing and query shape; templates are written for the shop
schema and pass the usual checks (four fixtures, solver, Haiku review, held-out
containment). Their verified solutions are appended to `solutions_v2.jsonl` before the
2B pipeline's second distillation stage reads it; the running RL stage is unchanged.

## MiniCPM5-2B pipeline (p2b)

`scripts/pipeline.sh ../models/MiniCPM5-2B p2b 6`, 25-26 October 2026, 19:20 to 06:19.
Stages and in-run scores (selection / check / reserved parameters):

| Stage | Run | Selection | Check | Reserved |
| --- | --- | --- | --- | --- |
| Released | p2b_sft (before) | 0.520 | 0.750 | 0.641 |
| Distillation, fresh adapter, 42 layers, 787 solutions, 50 updates | p2b_sft | 0.664 | 0.924 | 0.953 |
| GRPO, 23 steps in 6 h (time limit), selected step 23 | p2b_rl | 0.750 | 0.951-0.979 | 0.969 |
| Distillation, 884 solutions + 579 replayed answers, 92 updates | p2b_sft2 | 0.757 | 0.965 | 0.953 |

The second distillation's selection score was higher (0.757 against 0.750), so the
pipeline scored p2b_sft2/final. Final scoring (`release-2b-report.md`, 16 attempts,
384 tokens per turn, test exclusion as recorded): test 0.383 -> 0.731, check
0.736 -> 0.962, selection 0.474 -> 0.760, reserved parameters 0.695 -> 0.988,
attempt-weighted 0.525 -> 0.831. Outcomes: 1,223 correct, 212 wrong, 37 out of turns.

RL steps took 10-36 minutes; steps 2-3 slowed while template generation and solution
collection ran beside training (rollout 1,742 s at 28 tokens/s against 299 s at 172
tokens/s once they finished). Swap counters showed about 184 GB written since boot,
from overlapping jobs and a 7.5 GB `asitop` process earlier in the day; none during the
measured quiet period.
