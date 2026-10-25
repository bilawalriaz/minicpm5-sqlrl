import { RiAlarmWarningLine, RiFlagLine, RiTimerLine, RiTrophyLine } from "@remixicon/react";
import type { ReactNode } from "react";
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { type Stat, StatCards } from "@/components/application/dashboard/stat-cards";
import { Chip } from "@/components/base/badges/chip";
import type { CurriculumSnapshot, EvalRow, RunDetail, StepMetrics } from "@/lib/api";
import { cx } from "@/utils/cx";
import { OUTCOME_KEYS, OUTCOMES } from "./status";

interface Series {
  key: string;
  label: string;
  color: string;
  /** Raw per-step values are noisy; draw them thin behind a smoothed line. */
  faint?: boolean;
}

// Teacher columns are absent on runs without --teacher.
type Row = Record<string, number | undefined>;

const axisTick = { fontSize: 12, fill: "var(--color-text-tertiary)" };

/** Exponential moving average, so a trend shows through per-step noise. */
function ema(values: number[], alpha = 0.15) {
  let prev = values[0];
  return values.map((v) => (prev = alpha * v + (1 - alpha) * prev));
}

function ChartFrame({ title, headline, legend, children, className }: {
  title: string;
  headline?: ReactNode;
  legend: Series[];
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cx("flex h-[300px] min-w-0 flex-col gap-4 rounded-2xl bg-background-secondary-default px-4 pt-4 pb-3", className)}>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 flex-col gap-0.5">
          <h2 className="text-body-medium text-text-secondary">{title}</h2>
          {headline && <p className="text-title-3-medium text-text-primary tabular-nums">{headline}</p>}
        </div>
        <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-body-2-medium text-text-secondary">
          {legend.map((s) => (
            <li key={s.key} className="flex items-center gap-1.5">
              <span className="size-2 rounded-full" style={{ background: s.color }} aria-hidden />
              {s.label}
            </li>
          ))}
        </ul>
      </div>
      <div className="min-h-0 flex-1">{children}</div>
    </section>
  );
}

function ChartTooltip({ active, payload, label, series, format }: {
  active?: boolean;
  payload?: { dataKey?: string | number; value?: number }[];
  label?: number;
  series: Series[];
  format: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-popover px-3 py-2 text-body-2-regular text-text-primary shadow-dropdown">
      <p className="text-body-2-medium">Step {label}</p>
      {payload.map((p) => {
        const s = series.find((x) => x.key === p.dataKey);
        return s && p.value !== undefined ? (
          <p key={s.key} className="flex justify-between gap-4 tabular-nums">
            <span className="text-text-secondary">{s.label}</span>
            {format(p.value)}
          </p>
        ) : null;
      })}
    </div>
  );
}

function LineChartCard({ title, headline, data, series, format, domain, className }: {
  title: string;
  headline?: ReactNode;
  data: Row[];
  series: Series[];
  format: (v: number) => string;
  domain?: [number, number];
  className?: string;
}) {
  return (
    <ChartFrame title={title} headline={headline} legend={series.filter((s) => !s.faint)} className={className}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--color-chart-track)" />
          <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} tickLine={false} axisLine={false} tick={axisTick} />
          <YAxis width={44} domain={domain ?? ["auto", "auto"]} tickFormatter={format} tickLine={false} axisLine={false} tick={axisTick} />
          <Tooltip content={<ChartTooltip series={series} format={format} />} cursor={{ stroke: "var(--color-chart-cursor)", strokeDasharray: "4 4" }} />
          {series.map((s) => (
            <Line
              key={s.key}
              dataKey={s.key}
              stroke={s.color}
              strokeWidth={s.faint ? 1 : 2.5}
              strokeOpacity={s.faint ? 0.35 : 1}
              dot={data.length < 30 && !s.faint ? { r: 3 } : false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

/** Share of rollouts ending each way, per step. A growing "Out of turns" or "Hit token limit" band is a behaviour change worth reading. */
function OutcomeChart({ metrics, className }: { metrics: StepMetrics[]; className?: string }) {
  const data = metrics.map((m) => {
    const total = Object.values(m.outcomes).reduce((a, b) => a + b, 0);
    return Object.fromEntries([["step", m.step], ...OUTCOME_KEYS.map((k) => [k, (m.outcomes[k] ?? 0) / total])]);
  });
  const series = OUTCOME_KEYS.map((k) => ({ key: k, label: OUTCOMES[k].label, color: OUTCOMES[k].chart }));
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  return (
    <ChartFrame title="How rollouts end" legend={series} className={className}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} tickLine={false} axisLine={false} tick={axisTick} />
          <YAxis width={44} domain={[0, 1]} tickFormatter={pct} tickLine={false} axisLine={false} tick={axisTick} />
          <Tooltip content={<ChartTooltip series={series} format={pct} />} cursor={{ stroke: "var(--color-chart-cursor)", strokeDasharray: "4 4" }} />
          {series.map((s) => (
            <Area key={s.key} dataKey={s.key} stackId="1" stroke={s.color} fill={s.color} fillOpacity={0.55} isAnimationActive={false} />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

const BUCKETS = 8;

/** A rate per row (template or skill) over training, in equal step buckets, plus the first and last eval. */
function RateGrid({ title, rowLabel, metrics, rates, evals }: {
  title: string;
  rowLabel: string;
  metrics: StepMetrics[];
  rates: (byTemplate: Record<string, number>) => Record<string, number>;
  evals: EvalRow[];
}) {
  const size = Math.max(1, Math.ceil(metrics.length / BUCKETS));
  const buckets = Array.from({ length: Math.ceil(metrics.length / size) }, (_, i) => metrics.slice(i * size, (i + 1) * size));
  const perStep = metrics.map((m) => rates(m.by_template));
  // Generated templates appear partway through a run, so collect rows from every step.
  const rows = [...new Set(perStep.flatMap((r) => Object.keys(r)))].sort();
  const mean = (bucket: number, row: string) => {
    const vals = perStep.slice(bucket * size, (bucket + 1) * size).map((r) => r[row]).filter((v) => v !== undefined);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };
  const firstEval = evals[0] && rates(evals[0].by_template);
  const lastEval = evals.length > 1 ? rates(evals.at(-1)!.by_template) : undefined;

  return (
    <section className="flex flex-col gap-3 rounded-2xl bg-background-secondary-default p-4">
      <h2 className="text-body-medium text-text-secondary">{title}</h2>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] border-separate border-spacing-1 text-body-2-regular tabular-nums">
          <thead>
            <tr className="text-text-secondary">
              <th scope="col" className="text-start font-normal">{rowLabel}</th>
              {buckets.map((b) => (
                <th key={b[0].step} scope="col" className="font-normal">
                  {b[0].step}–{b.at(-1)!.step}
                </th>
              ))}
              {firstEval && <th scope="col" className="font-normal">Eval first</th>}
              {lastEval && <th scope="col" className="font-normal">Eval last</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row}>
                <th scope="row" className="text-start font-normal text-text-primary">
                  {row.replaceAll("_", " ")}
                </th>
                {buckets.map((b, i) => (
                  <Cell key={b[0].step} value={mean(i, row)} />
                ))}
                {firstEval && <Cell value={firstEval[row] ?? null} />}
                {lastEval && <Cell value={lastEval[row] ?? null} />}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/** Mean of each skill's templates. */
function skillRates(skillsOf: Record<string, string[]>) {
  return (byTemplate: Record<string, number>) => {
    const sums: Record<string, number[]> = {};
    for (const [template, rate] of Object.entries(byTemplate)) {
      for (const skill of skillsOf[template] ?? []) (sums[skill] ??= []).push(rate);
    }
    return Object.fromEntries(Object.entries(sums).map(([k, v]) => [k, v.reduce((a, b) => a + b, 0) / v.length]));
  };
}

/** What the sampler will draw next: each template's solve rate and its share of the next batch. */
function CurriculumTable({ snapshot, addedAt }: { snapshot: CurriculumSnapshot; addedAt: Record<string, number> }) {
  const rows = Object.entries(snapshot.templates).sort((a, b) => b[1].share - a[1].share);
  return (
    <section className="flex flex-col gap-3 rounded-2xl bg-background-secondary-default p-4">
      <h2 className="text-body-medium text-text-secondary">Curriculum: what the next steps sample</h2>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-body-2-regular tabular-nums">
          <thead>
            <tr className="text-text-secondary">
              <th scope="col" className="py-1 text-start font-normal">Template</th>
              <th scope="col" className="py-1 text-start font-normal">Skills</th>
              <th scope="col" className="py-1 text-end font-normal">Tier</th>
              <th scope="col" className="py-1 text-end font-normal">Solve rate</th>
              <th scope="col" className="py-1 text-end font-normal">Rollouts</th>
              <th scope="col" className="py-1 text-end font-normal">Share of next batch</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, t]) => (
              <tr key={name} className="border-t border-separator-border">
                <th scope="row" className="py-1.5 pe-3 text-start font-normal text-text-primary">
                  <span className="flex flex-wrap items-center gap-2">
                    {name.replaceAll("_", " ")}
                    {t.generated && <Chip color="blue">{addedAt[name] ? `Generated at step ${addedAt[name]}` : "Generated"}</Chip>}
                  </span>
                </th>
                <td className="py-1.5 pe-3 text-text-secondary">{t.skills.map((s) => s.replaceAll("_", " ")).join(", ")}</td>
                <td className="py-1.5 text-end">{t.tier}</td>
                <td className="py-1.5 text-end whitespace-nowrap">{t.rate === null ? "not seen" : pct(t.rate)}</td>
                <td className="py-1.5 text-end">{t.seen}</td>
                <td className="py-1.5 text-end">{pct(t.share)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Cell({ value }: { value: number | null }) {
  return (
    <td
      className="rounded-md px-2 py-1 text-center text-text-primary"
      style={value === null ? undefined : { background: `color-mix(in oklab, var(--color-chart-3) ${Math.round(Math.max(0, value) * 70)}%, var(--color-background-primary-default))` }}
    >
      {value === null ? "" : value.toFixed(2)}
    </td>
  );
}

const mean = (rows: StepMetrics[], pick: (m: StepMetrics) => number) => rows.reduce((a, m) => a + pick(m), 0) / rows.length;
const pct = (v: number) => `${Math.round(v * 100)}%`;
const fixed = (digits: number) => (v: number) => v.toFixed(digits);
const signed = (v: number, fmt: (v: number) => string) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${fmt(Math.abs(v))}`;

/** Everything on the Metrics tab. Lazy-loaded with recharts. */
export function MetricsPanel({ run }: { run: RunDetail }) {
  const m = run.metrics;
  const recent = m.slice(-10);
  const early = m.slice(0, 10);
  const last = m.at(-1)!;
  const firstEval = run.evals[0];
  const lastEval = run.evals.at(-1);

  const reward = mean(recent, (r) => r.reward_mean);
  const rewardChange = reward - mean(early, (r) => r.reward_mean);
  const literal = mean(recent, (r) => r.literal_frac);
  const literalChange = literal - mean(early, (r) => r.literal_frac);
  const evalChange = lastEval && firstEval ? lastEval.pass1 - firstEval.pass1 : 0;

  const stats: Stat[] = [
    {
      icon: RiTrophyLine,
      label: "Reward, last 10 steps",
      value: reward.toFixed(3),
      delta: signed(rewardChange, fixed(3)),
      deltaColor: rewardChange > 0 ? "lime" : rewardChange < 0 ? "rose" : "neutral",
    },
    {
      icon: RiFlagLine,
      label: "Evaluation pass@1",
      value: lastEval ? lastEval.pass1.toFixed(3) : "Not run",
      delta: lastEval === firstEval ? "baseline" : signed(evalChange, fixed(3)),
      deltaColor: evalChange > 0 ? "lime" : evalChange < 0 ? "rose" : "neutral",
    },
    {
      icon: RiAlarmWarningLine,
      label: "Hard-coded answers",
      value: pct(literal),
      delta: signed(literalChange, pct),
      // More hard-coding is the bad direction.
      deltaColor: literalChange > 0 ? "rose" : literalChange < 0 ? "lime" : "neutral",
    },
    {
      icon: RiTimerLine,
      label: "Seconds per step",
      value: String(Math.round(last.rollout_s + last.train_s)),
      delta: `${Math.round(last.gen_tps)} tok/s`,
      deltaColor: "neutral",
    },
  ];

  const rewardEma = ema(m.map((r) => r.reward_mean));
  const rows = m.map((r, i) => ({
    step: r.step,
    reward: r.reward_mean,
    rewardEma: rewardEma[i],
    turns: r.turns_mean,
    tokens: r.gen_tokens_mean,
    prose: r.prose_tokens_mean,
    literal: r.literal_frac,
    literalCorrect: r.literal_correct_frac,
    kept: r.groups_kept / r.groups_total,
    gradNorm: r.grad_norm,
    deadGroups: r.teacher_dead_groups,
    rescued: r.teacher_rescued,
  }));
  const evalRows = run.evals.map((e) => ({ step: e.step, pass1: e.pass1, ...Object.fromEntries(Object.entries(e.by_tier).map(([t, v]) => [`tier${t}`, v])) }));
  const tiers = Object.keys(firstEval?.by_tier ?? {});

  return (
    <div className="flex flex-col gap-4">
      <StatCards stats={stats} />
      <div className="grid gap-4 xl:grid-cols-2">
        <LineChartCard
          title="Training reward"
          headline={reward.toFixed(3)}
          data={rows}
          // Not answering (no query, or out of turns) scores below 0 (--no-answer-reward).
          domain={[-0.2, 1]}
          format={fixed(2)}
          series={[
            { key: "reward", label: "Per step", color: "var(--color-chart-2)", faint: true },
            { key: "rewardEma", label: "Smoothed", color: "var(--color-chart-2)" },
          ]}
        />
        <LineChartCard
          title="Evaluation pass@1"
          headline={lastEval?.pass1.toFixed(3)}
          data={evalRows}
          domain={[0, 1]}
          format={fixed(2)}
          series={[
            { key: "pass1", label: "All", color: "var(--color-chart-1)" },
            ...tiers.map((t, i) => ({ key: `tier${t}`, label: `Tier ${t}`, color: `var(--color-chart-${i + 3})` })),
          ]}
        />
        <OutcomeChart metrics={m} />
        <LineChartCard
          title="Hard-coded final queries"
          headline={pct(literal)}
          data={rows}
          domain={[0, 1]}
          format={pct}
          series={[
            { key: "literal", label: "Of graded answers", color: "var(--color-chart-6)" },
            { key: "literalCorrect", label: "Of correct answers", color: "var(--color-chart-5)" },
          ]}
        />
        <LineChartCard
          title="Turns per rollout"
          headline={last.turns_mean.toFixed(2)}
          data={rows}
          format={fixed(1)}
          series={[{ key: "turns", label: "Mean", color: "var(--color-chart-4)" }]}
        />
        <LineChartCard
          title="Tokens generated per rollout"
          headline={Math.round(last.gen_tokens_mean)}
          data={rows}
          format={(v) => String(Math.round(v))}
          series={[
            { key: "tokens", label: "All generated", color: "var(--color-chart-7)" },
            { key: "prose", label: "Outside SQL calls", color: "var(--color-chart-5)" },
          ]}
        />
        <LineChartCard
          title="Questions with learning signal"
          headline={`${last.groups_kept} of ${last.groups_total}`}
          data={rows}
          domain={[0, 1]}
          format={pct}
          series={[{ key: "kept", label: "Mixed rewards in group", color: "var(--color-chart-3)" }]}
        />
        <LineChartCard
          title="Gradient norm"
          headline={last.grad_norm.toFixed(3)}
          data={rows}
          format={fixed(2)}
          series={[{ key: "gradNorm", label: "Before clipping", color: "var(--color-chart-8)" }]}
        />
        {last.teacher_spent !== undefined && (
          <LineChartCard
            title={`Teacher rescues (${run.config.teacher})`}
            headline={`$${last.teacher_spent.toFixed(2)} spent`}
            data={rows}
            format={(v) => String(Math.round(v))}
            series={[
              { key: "deadGroups", label: "Questions all rollouts failed", color: "var(--color-chart-6)" },
              { key: "rescued", label: "Rescued with a teacher solution", color: "var(--color-chart-1)" },
            ]}
          />
        )}
      </div>
      {last.curriculum && (
        <>
          <CurriculumTable
            snapshot={last.curriculum}
            addedAt={Object.fromEntries(m.flatMap((r) => (r.new_templates ?? []).map((t) => [t, r.step])))}
          />
          <RateGrid
            title="Reward by skill"
            rowLabel="Skill"
            metrics={m}
            evals={run.evals}
            rates={skillRates(Object.fromEntries(Object.entries(last.curriculum.templates).map(([t, v]) => [t, v.skills])))}
          />
        </>
      )}
      <RateGrid title="Reward by question type" rowLabel="Template" metrics={m} evals={run.evals} rates={(byTemplate) => byTemplate} />
      <p className="text-body-2-regular text-text-secondary tabular-nums">Peak MLX memory last step: {last.peak_mem_gb.toFixed(1)} GB</p>
    </div>
  );
}
