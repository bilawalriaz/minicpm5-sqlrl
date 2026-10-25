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
