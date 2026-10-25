/** Typed access to build-time environment variables. */
export const env = {
  /** Base URL every API path is joined onto. `/api` goes through the Vite dev proxy to serve.py. */
  apiUrl: (import.meta.env.VITE_API_URL ?? "/api").replace(/\/$/, ""),
  /** Public origin of the deployed app, used for canonical URLs and OG tags. */
  siteUrl: (import.meta.env.VITE_SITE_URL ?? "").replace(/\/$/, ""),
} as const;
