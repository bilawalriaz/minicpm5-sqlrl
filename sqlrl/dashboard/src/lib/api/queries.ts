import { keepPreviousData, queryOptions } from "@tanstack/react-query";
import { api } from "./endpoints";

/** Training writes new rows every few seconds, so live views poll. */
const LIVE_MS = 3_000;

export const queries = {
  runs: () =>
    queryOptions({
      queryKey: ["runs"],
      queryFn: ({ signal }) => api.runs.list(signal),
      refetchInterval: LIVE_MS,
    }),

  run: (name: string) =>
    queryOptions({
      queryKey: ["runs", name],
      queryFn: ({ signal }) => api.runs.get(name, signal),
      refetchInterval: LIVE_MS,
    }),

  live: (name: string) =>
    queryOptions({
      queryKey: ["runs", name, "live"],
      queryFn: ({ signal }) => api.runs.live(name, signal),
      // The trainer rewrites the snapshot about once a second.
      refetchInterval: 1_000,
    }),

  samples: (name: string, step: number) =>
    queryOptions({
      queryKey: ["runs", name, "samples", step],
      queryFn: ({ signal }) => api.runs.samples(name, step, signal),
      // A step's samples never change once written.
      staleTime: Infinity,
      placeholderData: keepPreviousData,
    }),
};
