# Dashboard template

A fast, accessible admin dashboard frontend to start new projects from. Bring any backend.

- **Vite + React 19 + TypeScript**, TanStack Router (file-based, code-split) and TanStack Query
- **[coss ui](https://coss.com/ui/docs)** (Base UI) for primitives and **[BoardUI](https://www.boardui.com/components)** (React Aria) for dashboard blocks, sharing one Tailwind v4 theme with light and dark modes
- A typed API layer with a built-in **mock backend**, so it runs with no server
- Cookie or bearer auth, route guard, global 401 handling
- WCAG AA: skip link, landmarks, focus and route announcements on navigation, contrast checked with axe-core in both themes
- Pages: sign-in, overview (KPIs + charts), customers (URL-driven search, filter, sort, pagination, create/edit/delete), notifications, settings, ⌘K command palette

## Quick start

```bash
pnpm install
pnpm dev
```

Open http://localhost:5173 and sign in with `demo@example.com` / `password`.

## Use it for a new project

1. Create a repo from this template (GitHub: **Use this template**), or `gh repo create my-app --private --template bilawalriaz/react-dashboard-template`.
2. Set the app name in `.env` (`VITE_APP_NAME`) and `src/config/site.ts`, and swap `src/components/layout/logo.tsx` and `public/favicon.svg`.
3. Connect your backend: follow **[AGENTS.md → Connecting a backend](AGENTS.md#connecting-a-backend)**. The API contract is `src/lib/api/types.ts`; the mock in `src/lib/api/mock.ts` is a working reference implementation of it.

`AGENTS.md` is the full guide (architecture, API contract, auth, adding pages, UI and a11y rules, deployment). It is written for AI coding agents, and is just as useful for people.

## Scripts

| Command | What it does |
| --- | --- |
| `pnpm dev` | Dev server (mock API unless `VITE_API_MODE=http`) |
| `pnpm check` | Typecheck + lint |
| `pnpm build` | Production build to `dist/` |
| `pnpm preview` | Serve the production build |

## Why React + Vite (not Astro or Next.js)

- Both component libraries are React-only, so React is required either way.
- A dashboard is a stateful app shell behind a login: client-side routing, cached server state and instant navigation matter more than server rendering. A Vite SPA does this with the least machinery, and deploys as static files next to any backend.
- Astro shines for content sites, but its islands make shared app state (sidebar, command palette, query cache) awkward across pages. Use Astro for a separate public marketing site if you need content SEO.
- Next.js adds a server runtime and framework conventions this app doesn't need.
