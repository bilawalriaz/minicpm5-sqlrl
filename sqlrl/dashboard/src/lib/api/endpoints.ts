import { request } from "./client";
import type { Episode, LiveSnapshot, RunDetail, RunSummary } from "./types";

/** One function per backend endpoint (sqlrl/serve.py). */
export const api = {
  runs: {
    list: (signal?: AbortSignal) => request<RunSummary[]>("/runs", { signal }),
    get: (name: string, signal?: AbortSignal) => request<RunDetail>(`/runs/${encodeURIComponent(name)}`, { signal }),
    samples: (name: string, step: number, signal?: AbortSignal) =>
      request<Episode[]>(`/runs/${encodeURIComponent(name)}/samples/${step}`, { signal }),
    live: (name: string, signal?: AbortSignal) =>
      request<LiveSnapshot | null>(`/runs/${encodeURIComponent(name)}/live`, { signal }),
  },
};
