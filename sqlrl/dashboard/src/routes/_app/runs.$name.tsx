import { useQuery, useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { lazy, Suspense } from "react";
import { PageHeader } from "@/components/layout/page-header";
import { PageMeta } from "@/components/layout/page-meta";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { NumberField, NumberFieldDecrement, NumberFieldGroup, NumberFieldIncrement, NumberFieldInput } from "@/components/ui/number-field";
import { Select, SelectItem, SelectPopup, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsPanel, TabsTab } from "@/components/ui/tabs";
import { OUTCOME_KEYS, OUTCOMES, phaseLabel } from "@/features/runs/status";
import { LivePanel } from "@/features/runs/live";
import { GroupCard } from "@/features/runs/transcript";
import { type Episode, type Outcome, queries, type RunDetail } from "@/lib/api";
import { format } from "@/lib/format";

interface RunSearch {
  tab?: "live" | "rollouts";
  /** Live tab: hide rollouts that have finished. */
  active?: boolean;
  /** Undefined follows the newest step as training writes it. */
  step?: number;
  outcome?: Outcome;
  template?: string;
}

export const Route = createFileRoute("/_app/runs/$name")({
  validateSearch: (search: Record<string, unknown>): RunSearch => ({
    tab: search.tab === "rollouts" || search.tab === "live" ? search.tab : undefined,
    active: search.active === true || search.active === "true" ? true : undefined,
    step: Number(search.step) >= 1 ? Math.floor(Number(search.step)) : undefined,
    outcome: OUTCOME_KEYS.find((k) => k === search.outcome),
    template: typeof search.template === "string" && search.template ? search.template : undefined,
  }),
  loader: ({ context, params }) => context.queryClient.ensureQueryData(queries.run(params.name)),
  component: RunPage,
});

// recharts is ~110 KB gzipped; the header and rollouts tab shouldn't wait for it.
const MetricsPanel = lazy(() => import("@/features/runs/charts").then((m) => ({ default: m.MetricsPanel })));

function RunPage() {
  const { name } = Route.useParams();
  const search = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const { data: run } = useSuspenseQuery(queries.run(name));
  const latest = run.metrics.at(-1)?.step;

  return (
    <div className="flex flex-col gap-6">
      <PageMeta title={name} />
      <PageHeader title={name} description={<RunStatusLine run={run} />} />
      <Tabs
        value={search.tab ?? "metrics"}
        onValueChange={(tab) =>
          void navigate({ search: (s) => ({ ...s, tab: tab === "rollouts" || tab === "live" ? tab : undefined }) })
        }
      >
        <TabsList>
          <TabsTab value="metrics">Metrics</TabsTab>
          <TabsTab value="live">Live</TabsTab>
          <TabsTab value="rollouts">Rollouts</TabsTab>
        </TabsList>
        <TabsPanel value="metrics" className="pt-4">
          {latest === undefined ? (
            <NoStepsYet />
          ) : (
            <Suspense fallback={<Skeleton className="h-[640px] rounded-2xl" />}>
              <MetricsPanel run={run} />
            </Suspense>
          )}
        </TabsPanel>
        <TabsPanel value="live" className="pt-4">
          <LivePanel
            run={name}
            activeOnly={search.active ?? false}
            onActiveOnlyChange={(v) => void navigate({ search: (s) => ({ ...s, active: v || undefined }), replace: true })}
          />
        </TabsPanel>
        <TabsPanel value="rollouts" className="pt-4">
          {latest === undefined ? <NoStepsYet /> : <Rollouts run={run} latest={latest} />}
        </TabsPanel>
      </Tabs>
    </div>
  );
}

function RunStatusLine({ run }: { run: RunDetail }) {
  const c = run.config;
  return (
    <>
      {phaseLabel(run.status)}
      {run.status && `, updated ${format.relative(new Date(run.status.updated * 1000).toISOString())}`}
      <br />
      {c.steps} steps × {c.prompts} questions × {c.group} rollouts · lr {c.lr.toExponential()} · LoRA rank {c.rank} · temperature {c.temp} · tiers {c.tiers}
      {c.verifier_version && ` · ${c.verifier_version} / ${c.protocol_version}`}
      {c.hide_schema && " · schema hidden"}
      {c.teacher && ` · teacher ${c.teacher}`}
      {c.prose_penalty ? ` · prose penalty ${c.prose_penalty}/token` : ""}
    </>
  );
}

function NoStepsYet() {
  return (
    <Empty>
      <EmptyHeader>
        <EmptyTitle>No training steps yet</EmptyTitle>
        <EmptyDescription>The first step appears once its rollouts and gradient update finish.</EmptyDescription>
      </EmptyHeader>
    </Empty>
  );
}

const ALL = "all";

function Rollouts({ run, latest }: { run: RunDetail; latest: number }) {
  const search = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const step = Math.min(search.step ?? latest, latest);
  const { data: episodes, isPending } = useQuery(queries.samples(run.name, step));

  const templates = Object.keys(run.metrics[0].by_template).sort();
  const matches = (e: Episode) =>
    (!search.outcome || e.outcome === search.outcome) && (!search.template || e.template === search.template);
  const groups = new Map<number, Episode[]>();
  for (const e of episodes ?? []) groups.set(e.group, [...(groups.get(e.group) ?? []), e]);
  const shown = (episodes ?? []).filter(matches).length;

  const outcomeItems = [{ value: ALL, label: "All outcomes" }, ...OUTCOME_KEYS.map((k) => ({ value: k, label: OUTCOMES[k].label }))];
  const templateItems = [{ value: ALL, label: "All question types" }, ...templates.map((t) => ({ value: t, label: t.replaceAll("_", " ") }))];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end gap-3">
        <NumberField
          className="w-40"
          value={step}
          min={1}
          max={latest}
          onValueChange={(v) => v !== null && void navigate({ search: (s) => ({ ...s, step: v >= latest ? undefined : v }), replace: true })}
        >
          <NumberFieldGroup>
            <NumberFieldDecrement />
            <NumberFieldInput aria-label="Step" />
            <NumberFieldIncrement />
          </NumberFieldGroup>
        </NumberField>
        <Button variant="outline" disabled={search.step === undefined} onClick={() => void navigate({ search: (s) => ({ ...s, step: undefined }) })}>
          Follow latest step
        </Button>
        <Select
          items={outcomeItems}
          value={search.outcome ?? ALL}
          onValueChange={(v) => void navigate({ search: (s) => ({ ...s, outcome: OUTCOME_KEYS.find((k) => k === v) }) })}
        >
          <SelectTrigger aria-label="Filter by outcome" className="sm:w-56">
            <SelectValue />
          </SelectTrigger>
          <SelectPopup>
            {outcomeItems.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectPopup>
        </Select>
        <Select
          items={templateItems}
          value={search.template ?? ALL}
          onValueChange={(v) => void navigate({ search: (s) => ({ ...s, template: v === ALL || v === null ? undefined : v }) })}
        >
          <SelectTrigger aria-label="Filter by question type" className="sm:w-56">
            <SelectValue />
          </SelectTrigger>
          <SelectPopup>
            {templateItems.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectPopup>
        </Select>
        <p className="pb-2 text-body-2-regular text-text-secondary tabular-nums" aria-live="polite">
          Step {step}: {shown} of {episodes?.length ?? 0} rollouts
        </p>
      </div>

      {isPending ? (
        <Skeleton className="h-96 rounded-3xl" />
      ) : (
        <div className="flex flex-col gap-4">
          {[...groups].map(([group, g]) => g.some(matches) && <GroupCard key={group} episodes={g} shown={matches} />)}
        </div>
      )}
    </div>
  );
}
