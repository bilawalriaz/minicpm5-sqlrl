import { useQuery } from "@tanstack/react-query";
import { Chip } from "@/components/base/badges/chip";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { type LiveEpisode, queries } from "@/lib/api";
import { OUTCOMES } from "./status";
import { Turn } from "./transcript";

function LiveStatus({ episode }: { episode: LiveEpisode }) {
  if (episode.done && episode.outcome) {
    const outcome = OUTCOMES[episode.outcome];
    return <Chip color={outcome.chip}>{outcome.label}</Chip>;
  }
  if (episode.partial !== null) return <Chip color="blue">Writing turn {episode.turns + 1}</Chip>;
  return <Chip color="neutral">Waiting for turn {episode.turns + 1}</Chip>;
}

function LiveCard({ episode, index }: { episode: LiveEpisode; index: number }) {
  const segments = episode.partial ? [...episode.segments, { role: "assistant" as const, text: episode.partial }] : episode.segments;
  return (
    <article className="flex min-w-0 flex-col gap-2 rounded-2xl bg-background-secondary-default p-3">
      <div className="flex items-center gap-2">
        <span className="text-body-2-regular text-text-tertiary tabular-nums">#{index + 1}</span>
        <LiveStatus episode={episode} />
      </div>
      {/* column-reverse keeps the scroll position pinned to the newest text as it streams in */}
      <div className="flex max-h-72 flex-col-reverse overflow-y-auto">
        <ol className="flex flex-col gap-3">
          {segments.map((s, i) => (
            <Turn key={i} segment={s} />
          ))}
        </ol>
      </div>
    </article>
  );
}

/** Every rollout of the current batch, re-rendered as the trainer streams tokens. */
export function LivePanel({ run, activeOnly, onActiveOnlyChange }: {
  run: string;
  activeOnly: boolean;
  onActiveOnlyChange: (value: boolean) => void;
}) {
  const { data: live } = useQuery(queries.live(run));

  if (!live) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>Nothing streaming yet</EmptyTitle>
          <EmptyDescription>Rollouts appear here as soon as the trainer starts generating.</EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }

  const groups = new Map<number, { question: string; template: string; episodes: [LiveEpisode, number][] }>();
  for (const e of live.episodes) {
    const g = groups.get(e.group) ?? { question: e.question, template: e.template, episodes: [] };
    g.episodes.push([e, g.episodes.length]);
    groups.set(e.group, g);
  }
  const generating = live.episodes.filter((e) => !e.done).length;
  const correct = live.episodes.filter((e) => e.reward > 0).length;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-body-regular text-text-primary tabular-nums" aria-live="polite">
          {live.phase === "eval" ? "Held-out eval" : "Training rollouts"} at step {live.step}: {generating} of {live.episodes.length} still going,{" "}
          {correct} correct so far, {Math.round(live.seconds)}s in
        </p>
        <Label>
          <Switch checked={activeOnly} onCheckedChange={onActiveOnlyChange} />
          Only rollouts still going
        </Label>
      </div>
      {[...groups].map(([group, g]) => {
        const shown = g.episodes.filter(([e]) => !activeOnly || !e.done);
        if (!shown.length) return null;
        const solved = g.episodes.filter(([e]) => e.reward > 0).length;
        return (
          <section key={group} className="flex flex-col gap-3 rounded-3xl border border-separator-border p-3 sm:p-4">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <h3 className="min-w-0 flex-1 text-headline-semibold text-text-primary">{g.question}</h3>
              <span className="shrink-0 text-body-2-medium text-text-secondary tabular-nums">
                {solved}/{g.episodes.length} correct · {g.template.replaceAll("_", " ")}
              </span>
            </div>
            <div className="grid gap-2 sm:grid-cols-2 2xl:grid-cols-4">
              {shown.map(([e, i]) => (
                <LiveCard key={i} episode={e} index={i} />
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}
