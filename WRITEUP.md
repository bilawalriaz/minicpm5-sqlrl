# Teaching MiniCPM5 1B and 2B to be SQL agents on a MacBook

I've been training MiniCPM5-1B, and then MiniCPM5-2B, to answer questions about a SQLite database by writing and running SQL. Everything runs locally on an M3 Pro with 36 GB of memory, using MLX, LoRA adapters and group-relative policy optimisation (GRPO). A small API model helps generate training questions and check them.

The short version: on a test set of question types neither model trained on, written after every design decision and scored once, the 1B goes from 3.6% to 44.7% correct and the 2B from 38.3% to 73.1%. On the question types they did train on, measured on databases they had never seen, the 1B goes from 15% to 95% and the 2B from 70% to 99%.

## The task

The model gets a question such as "Which customers have placed more than 10 orders? Return only their names." It has two tools:

- `run_sql` runs a read-only query and shows ten preview rows.
- `submit_sql` submits the final query and ends the episode.

The complete result of the submitted query is the answer. Prose doesn't count, and the preview limit doesn't matter for grading. A correct answer scores 1, a wrong one 0, and running out of turns or tokens scores -0.2.

## Grading that can't be gamed

Most of the early work went into the verifier, because a reward model that accepts wrong SQL teaches wrong SQL.

Each submission is run against four databases and has to match the reference query on all of them. Row order is ignored, but column order, duplicate rows, text and integer precision all count. Results over 10,000 rows return an error instead of being silently cut off, and queries that read the clock are rejected.

The four databases aren't random copies. One is deliberately adversarial: two different customers with the same name in every city, two same-name customers who are both top spenders, prices exactly on the thresholds the questions use, signups and orders outside the usual years, and cancelled orders large enough to change which category "wins" if you forget the status filter.

I tested it with 21 plausible wrong rewrites of the reference queries, things like `COUNT(DISTINCT name)` instead of `COUNT(*)`, `<=` instead of `<`, `GROUP BY name` instead of `GROUP BY id`, and a missing `status = 'shipped'`. Eleven of them passed the previous version of the verifier for at least one question. All 21 now fail for every parameter, while correct rewrites (`NOT EXISTS`, `LEFT JOIN ... IS NULL`, `BETWEEN`) still pass.

Regrading the archive from the first training run with the stricter checks rejected 376 of 2,266 answers that had been marked correct.

## Measuring generalisation

An early run looked great: evaluation went from 25% to 82.5%. Then I noticed that 23 of the 25 evaluation questions had appeared in training. That number measured memory, not ability.

So the evaluation now uses separate databases and three kinds of question:

| Split | What it tests |
| --- | --- |
| Reserved parameters | Trained question types with parameter values and data the model never trained on |
| Selection set | Five never-trained question types, used to pick checkpoints and stop early |
| Check set | Five other never-trained question types, scored only at the start and end of a run |

A separate ten-template test set was added after all the design decisions. It gets scored exactly once, on the released model and the final checkpoint, with the exclusion rule written down before any result existed.

## Results

All figures compare against the released `openbmb/MiniCPM5-1B` (bf16), not an earlier base checkpoint. Each question was attempted 16 times at temperature 1 with up to 384 tokens per turn. The brackets are 95% intervals from resampling questions.

| | Released 1B | Fine-tuned 1B |
| --- | --- | --- |
| Test set: 9 never-trained types, 33 questions | 3.6% (1-7%) | 44.7% (34-56%) |
| Check set: 5 never-trained types, 18 questions | 12.2% (2-23%) | 70.1% (56-83%) |
| Selection set: 5 never-trained types, 19 questions | 5.6% (2-11%) | 50.0% (32-69%) |
| Trained types, unseen parameters and data | 14.8% (5-28%) | 94.9% (91-98%) |
| All attempts | 8.0% | 58.8% |

One test template was excluded under the rule written before scoring, because a generated training template asks almost the same thing. Per-template results are in `sqlrl/engineering/release-1b-report.md`.

The released model mostly fails on protocol: 640 of its 1,472 attempts never submit SQL, and another 507 run out of turns. The fine-tuned model submits on almost every attempt, so its remaining errors are wrong SQL. The hardest held-out types are still hard: counting groups that pass a `HAVING` filter, returned-order ratios and month-of-year aggregation sit between 0% and 19%. Some others went from 0% to around 90%.

## What stopped it improving, and the fixes

The interesting part of this project has been the ways that more training made the model worse.

It ran out of things to learn. By step 60 of the first long run the model solved nearly every training question, so most updates carried no signal. Held-out scores peaked around step 160 and then drifted down. The fix was more question types: an API model now proposes new templates whenever the learning signal drops, and every proposal goes through the same data checks as the hand-written ones.

New types crowded out old ones. Adding over a hundred harder templates made the model forget simple questions within four steps. It started inventing tables like `books` and `categories`. A fixed share of each batch now replays templates it has already mastered.

A fresh optimiser made big, noisy jumps. Restarting Adam on an already-trained adapter moves every weight by roughly the learning rate on the first few steps. Resuming the saved optimiser state fixed that, and resumed runs now take their learning rate from the command line, not from the saved state.

Wrong SQL was being rewarded. In groups where nobody got the answer right, a wrong submission (0) scored above "ran out of turns" (-0.2), so wrong SQL got a positive update. In one run that happened up to 21 times per step, and a measure of policy noise rose tenfold in eight steps. Wrong answers now never get positive credit.

Teacher examples from too far away caused damage. Injecting an API model's solutions to hard questions taught the student to bolt extra joins onto simple ones. Teacher solutions are now only used when the student already finds them reasonably likely (mean per-token negative log-likelihood under 1.0), and at half weight.

The policy swung between styles. With 16 questions per step from a varied pool, a few templates could dominate an update, and held-out scores moved 15-20 points within four steps. Evaluating and saving an exponential moving average of the weights, together with a lower learning rate and a larger batch, smoothed that out.

With all of that in place, a 20-step validation run moved the check set from 40% to 59% with no loss on the trained types. A longer run with twice the batch size then plateaued at about 45% on the selection set after 30 steps.

## Distillation beat more RL

RL only learns from questions the model sometimes gets right, so question types it can't solve yet give it nothing. To get past the plateau, GLM-5.3-flash wrote solutions for every training type, four parameter choices each. Every solution was checked against all four databases, leaving 787 verified solutions across 212 types. They cost $0.12.

The model then trained on those solutions directly, mixed with 567 of its own correct answers so it wouldn't forget what it already knew. That's 86 updates at a low learning rate. The selection set went from 46% to 51% and the check set from 65% to 72%, with trained types unchanged at 97%. That checkpoint is the final 1B model.

## Generating training data

The question bank has grown from 15 hand-written templates to 202 accepted generated ones. Claude Haiku 5.5 and GLM-5.3 propose templates, and a different model has to solve them without seeing the reference answer. In the latest generation round, Haiku also reviews each template for ambiguity. A template is rejected if its answer is the same on every database, if it depends on tie order, if a filter in it never changes the answer, or if its answers overlap a held-out question.

When the model keeps failing a template, the generator writes simpler stepping-stone templates that teach its parts. At 1B parameters, the model learns best from small steps it can almost manage.

Total API spend is about $2, most of it on template generation with review. All training and evaluation runs locally.

## Running the 1B model locally

`sqlrl/scripts/export_gguf.sh` fuses the final adapter into the model and writes BF16, Q8_0 (1.15 GB) and Q4_K_M (0.69 GB) GGUF files. It sets no-think mode as the chat template default, because that is how the model was trained.

`docs/lmstudio.md` explains how to test it in LM Studio, with a system prompt and ten example questions with their measured success rates. The model writes its own tool calls in the MiniCPM format, so you run each query against the evaluation database and paste the result back.

## MiniCPM5-2B

The released 2B went through `sqlrl/scripts/pipeline.sh` unattended: score the released model, distil from a fresh LoRA adapter on all 42 layers, up to six hours of GRPO with early stopping, a second distillation pass, then one scoring of the test set. The whole run took just under 11 hours.

| | Released 1B | Fine-tuned 1B | Released 2B | Fine-tuned 2B |
| --- | --- | --- | --- | --- |
| Test set: 9 never-trained types, 33 questions | 3.6% (1-7%) | 44.7% (34-56%) | 38.3% (26-51%) | 73.1% (62-84%) |
| Check set: 5 never-trained types | 12.2% | 70.1% | 73.6% | 96.2% (91-100%) |
| Selection set: 5 never-trained types | 5.6% | 50.0% | 47.4% | 76.0% (62-89%) |
| Trained types, unseen parameters and data | 14.8% | 94.9% | 69.5% | 98.8% (98-100%) |
| All attempts | 8.0% | 58.8% | 52.5% | 83.1% |

The 1B gained more points (41 against 35 on the test set) because it started near zero. The 2B removed more of its errors: 56% of its test-set misses against 43% for the 1B.

Each stage helped the 2B. Scores here are from the in-run evaluations (8 attempts per held-out question):

| Stage | Selection set | Check set | Trained types |
| --- | --- | --- | --- |
| Released 2B | 52.0% | 75.0% | 64.1% |
| After distillation (787 solutions, 50 updates) | 66.4% | 92.4% | 95.3% |
| After 23 steps of GRPO (six hours) | 75.0% | 95-98% | 96.9% |
| After the second distillation | 75.7% | 96.5% | 95.3% |

Distillation did most of the work. The 2B already follows the tool protocol, so the teacher solutions only had to teach SQL patterns, and they lifted every split in about an hour and a half. RL then added nine points on the selection set. The second distillation pass was level with RL, and the pipeline kept it because its selection score was higher by one attempt.

On the final scoring, 1,223 of the fine-tuned 2B's 1,472 attempts were correct, 212 were wrong SQL and 37 ran out of turns; none ended without a submission. The hardest types are still hard: the share of returned orders in a category (5%), the busiest month of a year (27%) and lines per order by status (27%). Counting groups that pass a `HAVING` filter, where the fine-tuned 1B managed 10-19%, reached 69-77% on the 2B.

Two differences between the runs matter when comparing them. The 1B's final checkpoint came from the development runs, with far more RL than the 2B's six hours. And the 2B's second distillation included 97 solutions for templates generated from real Spider train questions, added after the 1B was finished. Running the 1B through the same pipeline would make the comparison exact.

Code, tests and the engineering records are in this repository. Model weights, databases and checkpoints stay local. The detailed evidence for every number above is in `sqlrl/engineering/verification.md`.
