# AGENTS.md

> **This copy is the sqlrl training dashboard.** It differs from the template below:
> - The backend is `../serve.py` (stdlib Python). It reads `../runs/<name>/` written by `../train.py`; the contract is `src/lib/api/types.ts`.
> - Local-only tool: no auth, no login page, no mock backend. `pnpm dev` proxies `/api` to `API_PROXY_TARGET` (default `http://localhost:8000`).
> - Pages: `/` (runs) and `/runs/$name` (metrics and rollouts tabs; step and filters live in the URL).
> - Sections below about auth, the mock, customers, notifications and settings describe the template, not this app.

Instructions for AI coding agents (Claude Code, Codex, Cursor, …) working in a project created from this template. Read this whole file before changing anything. `CLAUDE.md` points here.

This repository is a **frontend-only** dashboard: a static single-page app (SPA) that talks to a backend over HTTP/JSON. Your usual job is one of:

1. **Integrate a backend** → read [Connecting a backend](#connecting-a-backend).
2. **Add pages or features** → read [Adding a page](#adding-a-page) and [UI rules](#ui-rules).
3. **Deploy** → read [Deployment](#deployment).

Before you finish any task, run `pnpm check` and `pnpm build`; both must pass with no errors. Follow [Code rules](#code-rules) strictly: less code is the goal.

---

## Stack

| Concern | Choice | Notes |
| --- | --- | --- |
| Build | Vite 8 + React 19 + TypeScript | No SSR, no Next.js. Output is static files in `dist/`. |
| Routing | TanStack Router (file-based) | Routes in `src/routes/`; `src/routeTree.gen.ts` is generated, never edit it. Each route is code-split automatically. |
| Server state | TanStack Query | All API data goes through query definitions in `src/lib/api/queries.ts`. |
| Styling | Tailwind CSS v4 | Single entry `src/index.css`. No `tailwind.config.*`; tokens live in CSS. |
| Primitives | **coss ui** (Base UI), in `src/components/ui/` | Buttons, forms, dialogs, menus, select, tabs, toast, command palette, table, sheet, etc. |
| Dashboard blocks | **BoardUI** (React Aria), in `src/components/{base,application,foundations}/` | Stat cards, chart cards (recharts), notification center, auth card, theme toggle, chips, avatars, pagination. |
| Icons | `@remixicon/react` (BoardUI) and `lucide-react` (inside coss components) | Use Remix icons in app code. |
| Lint | oxlint (with `jsx-a11y`) | `pnpm lint` |

Package manager: **pnpm**. Node ≥ 20.19.

### Commands

```bash
pnpm install        # install
pnpm dev            # dev server on http://localhost:5173 (mock API by default)
pnpm check          # typecheck + lint (run before finishing)
pnpm build          # typecheck + production build into dist/
pnpm preview        # serve dist/ locally
```

Demo login in mock mode: `demo@example.com` / `password`.

---

## Project map

```
src/
  main.tsx                 App bootstrap: QueryClient, router, global 401 handling
  index.css                THE stylesheet: imports BoardUI styles, bridges coss tokens onto them, a11y overrides
  config/
    env.ts                 Typed VITE_* env vars (apiUrl, apiMode, authStrategy, siteUrl)
    site.ts                App name, description, navigation (sidebar + command palette read this)
  lib/
    api/
      types.ts             ★ THE API CONTRACT: every request/response shape
      client.ts            fetch wrapper: base URL, JSON, auth header/cookies, timeout, ApiError
      endpoints.ts         ★ one function per backend endpoint (the only place paths live)
      queries.ts           TanStack Query definitions (keys + fetchers + cache policy)
      mock.ts              In-browser fake backend implementing the contract (dev/demo only)
    auth.ts                useCurrentUser / useSignIn / useSignOut
    format.ts              Intl formatters (currency, numbers, dates, relative time)
  routes/
    __root.tsx             Toast provider, route announcer, 404 + error pages
    login.tsx              Public sign-in page (indexable)
    _app.tsx               Authenticated shell + auth guard (pathless layout, not in URLs)
    _app/index.tsx         /              Overview: KPIs + charts
    _app/customers.tsx     /customers     CRUD table: URL-driven search/filter/sort/pagination
    _app/notifications.tsx /notifications
    _app/settings.tsx      /settings      Forms with server-side validation errors
  components/
    layout/                App shell pieces written for this template (sidebar, command menu, page header/meta/states, route announcer, logo)
    ui/                    coss ui components (vendored, see "Vendored components")
    base/ application/ foundations/   BoardUI components (vendored)
  features/customers/      Feature-specific components (dialog, status maps)
  styles/                  BoardUI theme.css (tokens), typography.css (type scale), globals.css
public/                    favicon, manifest, robots.txt (served as-is)
```

---

## Connecting a backend

The frontend never calls `fetch` directly. The chain is:

`component → useQuery(queries.x()) / useMutation(api.x.y) → endpoints.ts → client.ts request() → backend`

With `VITE_API_MODE=mock` (the default), `request()` is routed to `src/lib/api/mock.ts` instead of the network. **The mock is a reference implementation of the contract below**: when building or adapting the backend, read it for exact behaviour (status codes, validation, pagination maths).

### Step 1: point the app at the backend

Create `.env.local` (git-ignored; see `.env.example`):

```bash
VITE_API_MODE=http
VITE_API_URL=/api                     # relative = same origin (recommended)
VITE_AUTH_STRATEGY=cookie             # or "bearer"
API_PROXY_TARGET=http://localhost:8000  # dev only: Vite proxies /api/* here
```

- **Relative `VITE_API_URL` (recommended):** in dev, Vite proxies `/api/*` to `API_PROXY_TARGET` (see `vite.config.ts`). In production, serve the SPA and the API from the same origin (reverse proxy, or the backend serves `dist/`). Then cookies work with no CORS.
- **Absolute `VITE_API_URL`** (`https://api.example.com`): the backend must send CORS headers: `Access-Control-Allow-Origin: <exact frontend origin>` (not `*`), `Access-Control-Allow-Credentials: true` (cookie mode), and allow `Content-Type, Authorization` headers and `GET, POST, PUT, PATCH, DELETE`. Cross-site cookies also need `SameSite=None; Secure`.
- The dev proxy keeps the path as is: `/api/customers` is forwarded as `/api/customers`. If the backend has no `/api` prefix, add `rewrite: (p) => p.replace(/^\/api/, '')` to the proxy entry in `vite.config.ts`.

### Step 2: implement (or map) the contract

Types are in `src/lib/api/types.ts`. All bodies are JSON. Paths are relative to `VITE_API_URL`.

| Method | Path | Request | Success | Notes |
| --- | --- | --- | --- | --- |
| POST | `/auth/login` | `LoginInput` `{ email, password }` | `200` `LoginResponse` `{ user, token? }` | Cookie mode: set an httpOnly session cookie. Bearer mode: return `token`. Bad credentials → `401`. |
| POST | `/auth/logout` | none | `204` | Clear the session. |
| GET | `/auth/me` | none | `200` `User` | **`401` when not signed in.** The route guard relies on this. |
| GET | `/dashboard/overview` | none | `200` `Overview` `{ metrics[], revenue[12], orders[12] }` | `Metric.delta` is a fraction (`0.053` = +5.3%). `Metric.icon` keys map to icons in `routes/_app/index.tsx`. |
| GET | `/customers` | query: `page, pageSize, q?, status?, sort?, order?` | `200` `Paginated<Customer>` `{ items, total, page, pageSize }` | Page is 1-based. `sort` ∈ `name, mrr, createdAt`; `order` ∈ `asc, desc`. |
| POST | `/customers` | `CustomerInput` | `201` `Customer` | Validation → `422` with `fieldErrors`. |
| PATCH | `/customers/:id` | `Partial<CustomerInput>` | `200` `Customer` | `404` if missing. |
| DELETE | `/customers/:id` | none | `204` | |
| GET | `/notifications` | none | `200` `Notification[]` | Polled every 60 s. |
| POST | `/notifications/:id/read` | none | `204` | |
| POST | `/notifications/read-all` | none | `204` | |
| GET | `/settings` | none | `200` `Settings` | |
| PUT | `/settings` | `Settings` | `200` `Settings` | Validation → `422` with `fieldErrors`. |

**Errors.** Any non-2xx response should have a JSON body `ApiErrorBody`:

```json
{ "message": "Please fix the highlighted fields.", "code": "validation_error", "fieldErrors": { "email": "Enter a valid email address." } }
```

- `message` is shown to the user (toast or alert), so make it human-readable.
- `fieldErrors` keys must equal the form field `name`s; they render inline under each field.
- **`401` from any request** clears the cache and redirects to `/login?redirect=<current page>` (`src/main.tsx`). Don't return 401 for "forbidden"; use `403`.
- Dates are ISO 8601 strings. Money is in whole currency units (change `CURRENCY` in `src/lib/format.ts`).

**If the backend already exists with different paths or shapes,** do not change components. Adapt in `src/lib/api/endpoints.ts`: change the path, and/or map the response into the contract type, e.g.

```ts
list: async (params, signal) => {
  const res = await request<{ data: BackendCustomer[]; meta: { total: number } }>("/v2/clients", { query: { ...params }, signal });
  return { items: res.data.map(toCustomer), total: res.meta.total, page: params.page, pageSize: params.pageSize };
},
```

If the backend publishes OpenAPI, you may generate types (`pnpm dlx openapi-typescript <url> -o src/lib/api/schema.d.ts`) and re-export/map them in `types.ts`. Keep `types.ts` as the single source the UI imports from.

### Step 3: auth

- **Cookie (default, recommended):** backend sets `Set-Cookie: session=…; HttpOnly; Secure; SameSite=Lax; Path=/`. Requests use `credentials: "include"`. If the backend uses CSRF tokens, add the header in `request()` in `client.ts` (read it from a cookie or from a `/auth/csrf` call).
- **Bearer:** set `VITE_AUTH_STRATEGY=bearer`; `/auth/login` returns `{ user, token }`; the token is stored in `localStorage` (`app:token`) and sent as `Authorization: Bearer …`. For refresh tokens, extend `request()`: on `401`, call your refresh endpoint once and retry.
- **Third-party auth (Auth0, Clerk, Supabase, Cognito, …):** keep `useCurrentUser`/`useSignIn`/`useSignOut` in `src/lib/auth.ts` as the only API the UI uses. Implement them with the provider's SDK, inject the provider's access token in `request()`, and make `queries.me()` resolve to the `User` shape.
- Route protection is the `beforeLoad` in `src/routes/_app.tsx`. Every file under `src/routes/_app/` is protected automatically. Backend must still authorize every request; the frontend guard is UX only.

### Step 4: verify, then drop the mock

1. `pnpm dev`, sign in, and click through every page. Watch the network tab for non-2xx responses.
2. Search the code for remaining assumptions: `rg "mock" src`.
3. Optionally delete `src/lib/api/mock.ts` and the `import("./mock")` branch in `client.ts`. With `VITE_API_MODE=http` it is already excluded from the production bundle, so deleting it is optional.

---

## Adding a page

1. Create `src/routes/_app/<name>.tsx` (protected) or `src/routes/<name>.tsx` (public). The dev server regenerates `routeTree.gen.ts`.
2. Follow this shape:

```tsx
export const Route = createFileRoute("/_app/projects")({
  loader: ({ context }) => context.queryClient.ensureQueryData(queries.projects()), // prefetch = no loading spinner after navigation
  component: ProjectsPage,
});

function ProjectsPage() {
  const { data } = useSuspenseQuery(queries.projects());
  return (
    <div className="flex flex-col gap-6">
      <PageMeta title="Projects" />                        {/* <title>, description, robots */}
      <PageHeader title="Projects" description="…" actions={<Button>New project</Button>} />  {/* the page's only <h1> */}
      {/* content */}
    </div>
  );
}
```

3. Add it to `mainNav` (or `secondaryNav`) in `src/config/site.ts`, and extend the `NavItem["to"]` union. The sidebar, mobile drawer and ⌘K palette pick it up.
4. For a new resource: add types to `types.ts`, functions to `endpoints.ts`, a query to `queries.ts`, and a handler to `mock.ts` so the page works in mock mode too.
5. Mutations: `useMutation({ mutationFn: api.x.create, onSuccess: () => queryClient.invalidateQueries({ queryKey: ["x"] }) })`, then `toastManager.add({ type: "success", title: "…" })`. Show `ApiError.fieldErrors` with `<Form errors={…}>` (see `features/customers/customer-form-dialog.tsx`).
6. List/table state (page, search, filters, sort) goes in the URL via `validateSearch`, not `useState`, so it is shareable and survives reloads (see `routes/_app/customers.tsx`).

---

## UI rules

### Which library for what

- **Use coss ui (`@/components/ui/*`)** for interactive primitives: `Button`, `Input`, `InputGroup`, `Field`/`Form`, `Select`, `Combobox`, `Switch`, `Checkbox`, `Tabs`, `Dialog`, `AlertDialog`, `Sheet`, `Menu`, `Popover`, `Tooltip`, `Toast` (`toastManager`), `Command`, `Table`, `Card`, `Empty`, `Skeleton`, `Alert`, `Badge`, `Pagination`, `Calendar`. Docs: https://coss.com/ui/docs (each page has a `.md` version, e.g. `https://coss.com/ui/docs/components/select.md`; index at `https://coss.com/ui/llms.txt`). Base UI uses a **`render` prop instead of `asChild`**: `<Button render={<Link to="/x" />}>Go</Button>`.
- **Use BoardUI blocks** for dashboard visuals: `StatCards`, `RevenueChartCard`, `OrdersChartCard`, `NotificationCenter`, `AuthCard`, `ThemeToggle`, `Chip` (status pills), `Avatar`, `Pagination`, `Kbd`. Docs: https://www.boardui.com/components. More blocks: `npx boardui@latest list`, then `npx boardui@latest add <name>` (set `BOARDUI_TELEMETRY_DISABLED=1` to opt out of the CLI's anonymous telemetry). Many charts/templates are BoardUI **Pro** (paid); only add free ones unless the project has a licence.
- Don't hand-build something either library already has. Don't add a third component library.

### Styling

- Colours: use semantic tokens only, never raw palette classes (`text-gray-500`) or hex.
  - BoardUI tokens: `text-text-primary|secondary|tertiary`, `bg-background-full` (page), `bg-background-primary-default` (cards), `bg-background-secondary-default` (panels), `border-border-button-default`, `border-separator-border`, `text-foreground-icon-secondary`, `chart-1…9`, `accent-50…950` (brand ramp).
  - coss tokens: `bg-background`, `text-foreground`, `text-muted-foreground`, `bg-primary`, `border-border`, `ring-ring`, `bg-card`, `text-destructive`… These are **bridged** onto BoardUI tokens in `src/index.css`, so either set is fine and both flip in dark mode.
- **Re-brand:** change the `--color-accent-*` ramp in `src/styles/theme.css` (BoardUI) and both libraries follow. Keep white-on-`accent-600` at ≥ 4.5:1 contrast.
- Typography: BoardUI composite utilities set size + weight + line-height together: `text-title-2-medium` (page h1), `text-title-3-medium` (section), `text-headline-semibold`, `text-body-medium`, `text-body-regular`, `text-body-2-regular`, `text-caption-1-medium`. Don't stack `text-sm font-medium leading-5`.
- Merge classes with `cx()` from `@/utils/cx` in app/BoardUI code (`cn()` from `@/lib/utils` inside coss components).
- Cards/panels: `rounded-2xl`/`rounded-3xl`; inputs and menu rows: `rounded-md`–`rounded-xl`; pills: `rounded-full`.
- Dark mode is the `.dark` class on `<html>`. Don't write `dark:` overrides with raw colours; pick a token.
- Use logical properties (`ms-`, `pe-`, `start-`, `text-start`) so RTL works.

### Accessibility (WCAG 2.2 AA)

Already handled globally, keep it working: skip link, landmarks (`aside`/`nav`/`header`/`main#main`), focus moves to `<main>` and the new title is announced after every navigation (`route-announcer.tsx`), visible `:focus-visible` rings, `prefers-reduced-motion` respected, colour contrast overrides in `index.css` (verified with axe-core in both themes).

When you write UI:
- One `<h1>` per page (use `PageHeader`); headings in order.
- Every input has a visible label (`FieldLabel`) or `aria-label`; icon-only buttons need `aria-label`; decorative icons get `aria-hidden`.
- Use real `<button>`/`<a>`/`<Link>`; never clickable `<div>`s.
- Don't convey state by colour alone (status chips have text).
- Tables: `<th>` headers, `aria-sort` on sortable columns.
- Test keyboard-only: Tab through the page, open/close every dialog/menu with Enter/Escape.
- Optional audit: with `pnpm dev` running, inject `/node_modules/axe-core/axe.min.js` in the browser console and run `await axe.run()`.

### Performance

- Every route is code-split automatically. Keep heavy libraries out of `_app.tsx`/`__root.tsx` (they load on every page).
- Lazy-load heavy widgets with `React.lazy` + `Suspense` (charts already are: recharts is ~110 KB gzipped).
- Prefetch data in the route `loader` with `ensureQueryData`; links preload on hover (`defaultPreload: "intent"`).
- Don't put API data in `useState`/context; TanStack Query is the cache.
- Images: add `width`/`height`, `loading="lazy"`, and prefer WebP/AVIF in `public/`.

### SEO

A dashboard behind login should not be indexed. Defaults:
- `index.html` and `<PageMeta>` emit `noindex` except on pages that pass `<PageMeta index />` (only `/login`).
- `public/robots.txt` allows `/login`, disallows everything else.
- Set `VITE_SITE_URL` in production for canonical URLs and Open Graph tags.
- React 19 hoists `<title>`/`<meta>`/`<link>` from `PageMeta` into `<head>`; no helmet library.
- **If the product needs public, indexable marketing pages**, build them as a separate static site (e.g. Astro) or prerender them; a client-rendered SPA is the wrong tool for content SEO. Keep this app for the authenticated product.

---

## Code rules

The aim is the smallest, plainest change that does the job. A reviewer should be able to read the diff once and see nothing they'd delete.

### Write less

- Solve the task in front of you. No speculative abstractions, generic helpers, config options, props or "for later" hooks that nothing uses yet.
- Don't refactor, rename or reformat code you weren't asked to touch. Keep the diff focused.
- Prefer editing an existing file over creating a new one. No one-off `helpers.ts`/`utils2.ts`, no new barrel files.
- Don't add a dependency for something the platform, an installed library or ~20 lines can do. Ask before adding any new runtime dependency.
- Delete what you replace: no dead code, commented-out code, unused exports, leftover `console.log`, or "legacy" fallbacks.

### No defensive programming

- Trust the types and your own code. Don't null-check, `?.`, or `?? default` values the types already guarantee.
- No `try/catch` unless you handle a specific, expected failure in a specific way. Let unexpected errors throw: the route `errorComponent`, React Query's error state and the global 401 handler already deal with them. Don't catch-and-log, catch-and-rethrow, or catch-and-return-null.
- Validate only at trust boundaries: URL search params (`validateSearch`), user input (forms), and responses from a backend whose shape you don't control (map them in `endpoints.ts`, once). Not between internal functions.
- No fallback values that hide bugs (`|| []`, `|| 0`, `|| "Unknown"`) unless "missing" is a real, expected state of the data.
- No `any`, no `as` casts to silence the compiler, no `@ts-ignore`/`@ts-expect-error`, no lint-disable comments. Fix the type instead.
- No retries, timeouts, caches or loading flags of your own: `client.ts` and TanStack Query already provide them.

### React

- Derive values during render; don't mirror props or query data into `useState`, and don't sync state with `useEffect`. Effects are only for talking to things outside React (timers, event listeners, the DOM).
- No `useMemo`/`useCallback`/`memo` by default. Add them only for a measured problem or a stable dependency an effect needs.
- Server data lives in TanStack Query only, never copied into context or global stores.

### Comments and naming

- Comment *why* (a constraint, a workaround, a non-obvious decision), never *what* the next line does. Most code needs no comment.
- No changelog comments ("Added X", "Updated to use Y", "Fixed bug"), no banner/section-divider comments, no TODOs without a linked issue.
- Names say what a thing is in the domain (`unpaidInvoices`), not its type or role (`dataArray`, `handleData2`).

### Tests: minimal

- There is no test suite by default, and that's deliberate. Don't add tests unless asked.
- If you add some, only for pure, non-trivial logic where a bug would be silent (formatters, data mapping in `endpoints.ts`, calculations). Use Vitest (`pnpm add -D vitest`), colocate as `*.test.ts`, keep it to a few cases that each check a real behaviour.
- Never: snapshot tests, tests of vendored components or plain markup, tests that restate the implementation, or heavy mocking of React/router internals.
- For UI work, verification is `pnpm check`, `pnpm build`, and actually using the feature in the browser (mock mode), in light and dark, keyboard and 375 px.

### UI copy and design

- Plain, specific copy: "Customer deleted", "Couldn’t save settings". No marketing tone, exclamation marks, emoji, or apologies.
- No decorative filler: no intro paragraphs restating the heading, no badges/eyebrows/"New!" labels, no gradient text, glows or glassmorphism, no placeholder stats. Use the existing tokens and components; don't invent new visual styles.

### Reporting

- Don't create summary or notes files (`CHANGES.md`, `IMPLEMENTATION.md`, …). Update this file only when a project convention changes.
- When you finish, say briefly what changed and what you did *not* verify.

---

## Vendored components

`src/components/ui/` (coss ui) and `src/components/{base,application,foundations}/` (BoardUI) are source code copied in by their CLIs. You own them and may edit them, but prefer composing over editing. To add more:

```bash
pnpm dlx shadcn@latest add @coss/<component>     # coss ui (registry configured in components.json)
npx boardui@latest add <name>                    # BoardUI
```

Re-adding an existing component **overwrites local edits**. These files have been modified for this template; re-apply the change if you re-add them:

| File | Change |
| --- | --- |
| `application/theme/theme-toggle.tsx` | Defaults to the OS theme when nothing is stored (was always light). Pre-paint script in `index.html` mirrors this. |
| `application/auth/auth-card.tsx` | Added `pending`, `forgotHref`, `switchHref={null}` support (hide links/disable submit). |
| `application/notification-center/notification-center.tsx` | Added `onMarkAllRead`; unread dot has `role="img"`. |
| `ui/tabs.tsx` | Inactive tab text at full `muted-foreground` for contrast. |
| `lib/composer-editor.ts` | Removed an unused type import (strict TS). |

BoardUI demo blocks with hard-coded sample content (dashboard sidebar, team/user menus, settings modal, data-table example) were removed; `components/layout/app-sidebar.tsx` is the data-driven replacement for the BoardUI sidebar.

---

## Deployment

`pnpm build` produces static files in `dist/`. Any static host works. Requirements:

1. **SPA fallback:** unknown paths must serve `index.html` (client-side routing).
2. **Caching:** `dist/assets/*` are content-hashed → `Cache-Control: public, max-age=31536000, immutable`. `index.html` → `no-cache`.
3. **API:** either reverse-proxy `/api` to the backend on the same origin (recommended), or build with an absolute `VITE_API_URL` and configure CORS.
4. **Env vars are baked in at build time** (`VITE_*`). Set them in the build environment, not at runtime.

nginx:

```nginx
location /api/ { proxy_pass http://backend:8000; proxy_set_header Host $host; }
location /assets/ { add_header Cache-Control "public, max-age=31536000, immutable"; try_files $uri =404; }
location / { add_header Cache-Control "no-cache"; try_files $uri /index.html; }
```

Caddy: `handle /api/* { reverse_proxy backend:8000 }` and `handle { root * /srv/dist; try_files {path} /index.html; file_server }`.

Netlify / Cloudflare Pages: add `public/_redirects` with `/api/*  https://api.example.com/api/:splat  200` and `/*  /index.html  200`.

Backend serving the files itself (FastAPI, Express, Go, Rails, Laravel, …): mount `dist/` as static files, register API routes under `/api`, and add a catch-all GET that returns `dist/index.html` for any other non-file path.

---

## Definition of done

- [ ] `pnpm check` passes (types + lint).
- [ ] `pnpm build` passes.
- [ ] New pages: `PageMeta` + `PageHeader`, nav entry in `site.ts`, data via `queries.ts`, mock handler added.
- [ ] Works in light and dark, at 375 px wide (no horizontal scroll) and on desktop.
- [ ] Keyboard-only walkthrough works; no new axe violations.
- [ ] No secrets in `VITE_*` vars or committed `.env` files.
- [ ] Diff follows [Code rules](#code-rules): no unused code, no defensive checks, no needless tests, no new files or dependencies that weren't required.
