import type { ChipProps } from "@/components/base/badges/chip";
import type { Outcome, RunStatus } from "@/lib/api";

/** The trainer rewrites status.json at least once per rollout turn, so silence this long means it stopped. */
const STALE_AFTER_S = 300;

export function isStale(status: RunStatus) {
  return status.phase !== "done" && status.phase !== "error" && Date.now() / 1000 - status.updated > STALE_AFTER_S;
}

export function phaseLabel(status: RunStatus | null) {
  if (!status) return "No status yet";
  if (isStale(status)) return `Stopped at step ${status.step}`;
  switch (status.phase) {
    case "loading":
      return "Loading model";
    case "rollout":
      return `Step ${status.step}: rollouts, turn ${status.turn ?? 0}`;
    case "train":
      return `Step ${status.step}: gradient update`;
    case "eval":
      return `Evaluating at step ${status.step}`;
    case "error":
      return `Stopped at step ${status.step}: ${status.error}`;
    case "done":
      return status.stop_reason === "time limit" ? `Stopped at the time limit, ${status.step} steps` : `Finished, ${status.step} steps`;
  }
}

export const OUTCOMES: Record<Outcome, { label: string; chip: NonNullable<ChipProps["color"]>; chart: string }> = {
  correct: { label: "Correct", chip: "lime", chart: "var(--color-chart-3)" },
  wrong: { label: "Wrong answer", chip: "rose", chart: "var(--color-chart-6)" },
  no_query: { label: "No SQL answer submitted", chip: "purple", chart: "var(--color-chart-8)" },
  length: { label: "Hit token limit", chip: "yellow", chart: "var(--color-chart-4)" },
  max_turns: { label: "Out of turns", chip: "cyan", chart: "var(--color-chart-2)" },
};

export const OUTCOME_KEYS = Object.keys(OUTCOMES) as Outcome[];
