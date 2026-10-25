/**
 * The API contract, matching what sqlrl/serve.py returns from the files the
 * trainer writes under runs/<name>/. Times are Unix seconds (Python time.time()).
 */

export type Phase = "loading" | "rollout" | "train" | "eval" | "done" | "error";

export interface RunStatus {
  phase: Phase;
  step: number;
  updated: number;
  error?: string;
  /** Done phase only: "completed" or "time limit". */
  stop_reason?: string;
  /** Rollout phase only. */
  turn?: number;
  active?: number;
  gen_tokens?: number;
  seconds?: number;
}

export interface RunSummary {
  name: string;
  started: number;
  steps: number;
  step: number;
  reward: number | null;
  evalFirst: number | null;
  evalLast: number | null;
  status: RunStatus | null;
}

export type Outcome = "correct" | "wrong" | "no_query" | "length" | "max_turns";

export interface StepMetrics {
  step: number;
  time: number;
  reward_mean: number;
  outcomes: Partial<Record<Outcome, number>>;
  by_template: Record<string, number>;
  by_tier: Record<string, number>;
  turns_mean: number;
  /** Share of graded answers whose final query reads no table (e.g. SELECT 11). */
  literal_frac: number;
  /** Share of correct answers that were such literals. */
  literal_correct_frac: number;
  gen_tokens_mean: number;
  /** Tokens outside SQL calls (apologies, narration, the reply). Absent on runs from before the prose penalty. */
  prose_tokens_mean?: number;
  groups_kept: number;
  groups_total: number;
  loss: number;
  grad_norm: number;
  /** Absent on runs from before sql-v2.2. Those runs report mean reward in by_template and by_tier, not solve rates. */
  update_skipped_nonfinite?: boolean;
  train_tokens: number;
  rollout_s: number;
  train_s: number;
  gen_tps: number;
  peak_mem_gb: number;
  /** Present only when training with --teacher. */
  teacher_dead_groups?: number;
  teacher_rescued?: number;
  /** Dollars spent against the OpenCode Go allowance so far this run. */
  teacher_spent?: number;
  /** Absent on runs from before the adaptive curriculum. */
  curriculum?: CurriculumSnapshot;
  /** Templates generated for weak skills and accepted at the start of this step. */
  new_templates?: string[];
  evolve_spent?: number;
}

export interface CurriculumSnapshot {
  templates: Record<string, {
    /** Moving average of the student's solve rate; null until first sampled. */
    rate: number | null;
    seen: number;
    /** Probability of being sampled for the next step. */
    share: number;
    skills: string[];
    tier: number;
    generated: boolean;
  }>;
  skills: Record<string, number>;
}

export interface EvalRow {
  step: number;
  time: number;
  pass1: number;
  by_template: Record<string, number>;
  by_tier: Record<string, number>;
  /** Solve rate per evaluation split: heldout_template, reserved_params, shared_wording. Absent before sql-v2.2. */
  by_split?: Record<string, number>;
  outcomes: Partial<Record<Outcome, number>>;
  seconds: number;
  /** Check rows only: which checkpoint was scored, e.g. "initial_step_00000" or "selected_step_00120". */
  checkpoint?: string | null;
}

export interface RunConfig {
  name: string;
  verifier_version?: string;
  protocol_version?: string;
  steps: number;
  prompts: number;
  group: number;
  lr: number;
  rank: number;
  temp: number;
  max_turns: number;
  tiers: string;
  hide_schema: boolean;
  teacher: string | null;
  prose_penalty?: number;
  started: number;
}

export interface RunDetail {
  name: string;
  config: RunConfig;
  status: RunStatus | null;
  metrics: StepMetrics[];
  evals: EvalRow[];
  /** Check-set evaluations, from eval-v3: at the start, then on the final and selected checkpoints. */
  checks: EvalRow[];
  /** Written when training ends: the checkpoint with the best smoothed selection score. */
  selected: { step: number; smoothed_selection_score: number } | null;
}

export interface Segment {
  role: "assistant" | "tool";
  text: string;
}

type Cell = string | number | null;

export interface Episode {
  group: number;
  template: string;
  tier: number;
  question: string;
  gold_sql: string;
  reward: number;
  advantage: number;
  outcome: Outcome;
  turns: number;
  gen_tokens: number;
  prose_tokens?: number;
  segments: Segment[];
  /** Explicit submitted SQL in v2; last successful query in legacy runs. */
  final_sql: string | null;
  verifier_reason?: string;
  literal: boolean;
  answer_rows: Cell[][] | null;
  gold_rows: Cell[][] | null;
  /** "teacher" marks a verified API-model solution injected into an all-fail group. */
  source: "student" | "teacher";
}

/** Rollouts in flight, rewritten by the trainer about once a second (runs/<name>/live.json). */
export interface LiveSnapshot {
  phase: "rollout" | "eval";
  step: number;
  updated: number;
  /** Seconds since this rollout batch started. */
  seconds: number;
  episodes: LiveEpisode[];
}

export interface LiveEpisode {
  group: number;
  template: string;
  question: string;
  turns: number;
  segments: Segment[];
  /** Text of the turn being generated right now; null between turns and when done. */
  partial: string | null;
  done: boolean;
  outcome: Outcome | null;
  reward: number;
}

export interface ApiErrorBody {
  message: string;
}
