# Teaching a 1B model to be a SQL agent on a MacBook

I've been training MiniCPM5-1B to answer questions about a SQLite database by writing and running SQL. Everything runs locally on an M3 Pro with 36 GB of memory, using MLX, LoRA adapters and group-relative policy optimisation (GRPO). A small API model helps generate training questions and check them.

This is a write-up of where it's got to. The short version: on question types it never trained on, the released 1B model goes from 13% to roughly 55-59% correct. On the question types it did train on, it goes from under 15% to 94-100%, measured on databases it has never seen.

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

## Results so far

All figures compare against the released `openbmb/MiniCPM5-1B` (bf16), not an earlier base checkpoint.

| | Released 1B | Fine-tuned 1B |
| --- | --- | --- |
| Never-trained types, check set | 13% | 54-59% across the latest checkpoints |
| Never-trained types, selection set | 5% | 45% |
| Trained types, unseen data | 6% | 94-100% |
| All evaluation questions | 7% | about 61% |

The check set is small (18 questions from five templates), so treat the exact percentage loosely. The test-set numbers, with confidence intervals, will replace these once training finishes.

## What stopped it improving, and the fixes

The interesting part of this project has been the ways that more training made the model worse.

It ran out of things to learn. By step 60 of the first long run the model solved nearly every training question, so most updates carried no signal. Held-out scores peaked around step 160 and then drifted down. The fix was more question types: an API model now proposes new templates whenever the learning signal drops, and every proposal goes through the same data checks as the hand-written ones.

New types crowded out old ones. Adding over a hundred harder templates made the model forget simple questions within four steps. It started inventing tables like `books` and `categories`. A fixed share of each batch now replays templates it has already mastered.

A fresh optimiser made big, noisy jumps. Restarting Adam on an already-trained adapter moves every weight by roughly the learning rate on the first few steps. Resuming the saved optimiser state fixed that, and resumed runs now take their learning rate from the command line, not from the saved state.

Wrong SQL was being rewarded. In groups where nobody got the answer right, a wrong submission (0) scored above "ran out of turns" (-0.2), so wrong SQL got a positive update. In one run that happened up to 21 times per step, and a measure of policy noise rose tenfold in eight steps. Wrong answers now never get positive credit.

Teacher examples from too far away caused damage. Injecting an API model's solutions to hard questions taught the student to bolt extra joins onto simple ones. Teacher solutions are now only used when the student already finds them reasonably likely (mean per-token negative log-likelihood under 1.0), and at half weight.

The policy swung between styles. With 16 questions per step from a varied pool, a few templates could dominate an update, and held-out scores moved 15-20 points within four steps. Evaluating and saving an exponential moving average of the weights, together with a lower learning rate and a larger batch, smoothed that out.

With all of that in place, a 20-step validation run moved the check set from 40% to 59% with no loss on the trained types. That's the configuration the current long run uses.

## Generating training data

The question bank has grown from 15 hand-written templates to over 160 accepted generated ones. Claude Haiku 5.5 and GLM-5.3 propose templates, and a different model has to solve them without seeing the reference answer. In the latest generation round, Haiku also reviews each template for ambiguity. A template is rejected if its answer is the same on every database, if it depends on tie order, if a filter in it never changes the answer, or if its answers overlap a held-out question.

When the model keeps failing a template, the generator writes simpler stepping-stone templates that teach its parts. At 1B parameters, the model learns best from small steps it can almost manage.

Total API spend so far is under $1. All training and evaluation runs locally.

## What's next

The current run continues until the selection score stops improving. After that I'll try supervised distillation on verified solutions across the whole bank before another round of RL, which should teach new question types faster than RL can bootstrap them.

Then the same pipeline goes onto the released MiniCPM5-2B, so I can compare how much each model gains from its own starting point and how the two finished models compare.

Code, tests and the engineering records are in this repository. Model weights, databases and checkpoints stay local. The detailed evidence for every number above is in `sqlrl/engineering/verification.md`.
