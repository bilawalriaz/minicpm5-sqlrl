import { env } from "@/config/env";
import type { ApiErrorBody } from "./types";

/** Minimal fetch wrapper: base URL, JSON, timeout, typed errors. */

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, body: Partial<ApiErrorBody>) {
    super(body.message || `Request failed (${status})`);
    this.name = "ApiError";
    this.status = status;
  }
}

export const isNotFound = (error: unknown) => error instanceof ApiError && error.status === 404;

export async function request<T>(path: string, { signal }: { signal?: AbortSignal } = {}): Promise<T> {
  const timeout = AbortSignal.timeout(15_000);
  const res = await fetch(`${env.apiUrl}${path}`, {
    headers: { Accept: "application/json" },
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
  });
  const isJson = res.headers.get("content-type")?.includes("application/json");
  const payload: unknown = isJson ? await res.json() : await res.text();
  if (!res.ok) {
    throw new ApiError(res.status, typeof payload === "object" && payload ? payload : { message: String(payload) });
  }
  return payload as T;
}
