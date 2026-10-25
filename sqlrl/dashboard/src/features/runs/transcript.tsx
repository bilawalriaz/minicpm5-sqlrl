import { RiArrowRightSLine } from "@remixicon/react";
import { Chip } from "@/components/base/badges/chip";
import { Collapsible, CollapsiblePanel, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { Episode, Segment } from "@/lib/api";
import { cx } from "@/utils/cx";
import { OUTCOMES } from "./status";

// A call cut off by the token limit has no closing tag, so the end of text closes it too.
const CALL = /<function name="([^"]+)">([\s\S]*?)(?:<\/function>|$)/;
const PARAM = /<param name="query">(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?(?:<\/param>|$)/;

/** Split an assistant turn into the prose around its function call and the call itself. */
function parseTurn(text: string) {
  const m = CALL.exec(text);
  if (!m) return { before: text.trim(), call: null, after: "" };
  return {
    before: text.slice(0, m.index).trim(),
    call: { name: m[1], query: PARAM.exec(m[2])?.[1].trim() ?? m[2] },
    after: text.slice(m.index + m[0].length).trim(),
  };
}

const KEYWORDS =
  /\b(SELECT|FROM|WHERE|JOIN|LEFT|INNER|OUTER|ON|GROUP|BY|ORDER|HAVING|LIMIT|AS|AND|OR|NOT|IN|IS|NULL|LIKE|DISTINCT|COUNT|SUM|AVG|MIN|MAX|ROUND|COALESCE|CASE|WHEN|THEN|ELSE|END|DESC|ASC|UNION|EXCEPT|INTERSECT|EXISTS|BETWEEN)\b/gi;
const TOKENS = new RegExp(`('(?:[^']|'')*')|${KEYWORDS.source}|(\\b\\d+(?:\\.\\d+)?\\b)`, "gi");

function Sql({ sql }: { sql: string }) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const m of sql.matchAll(TOKENS)) {
    parts.push(sql.slice(last, m.index));
    const cls = m[1] ? "text-status-lime-text" : m[2] ? "text-status-blue-text font-medium" : "text-status-orange-text";
    parts.push(
      <span key={m.index} className={cls}>
        {m[0]}
      </span>,
    );
    last = m.index + m[0].length;
  }
  parts.push(sql.slice(last));
  return (
    <pre className="overflow-x-auto rounded-xl bg-background-tertiary-default px-3 py-2 font-mono text-body-2-regular whitespace-pre-wrap text-text-primary">
      <code>{parts}</code>
    </pre>
  );
}

type Cell = string | number | null;

function RowsTable({ rows, caption }: { rows: Cell[][]; caption?: string }) {
  if (!rows.length) return <p className="text-body-2-regular text-text-secondary">0 rows</p>;
  return (
    <div className="overflow-x-auto">
      <table className="text-body-2-regular tabular-nums">
        {caption && <caption className="pb-1 text-start text-body-2-medium text-text-secondary">{caption}</caption>}
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-separator-border last:border-0">
              {row.map((v, j) => (
                <td key={j} className="py-1 pe-4 text-text-primary">
                  {v === null ? <span className="text-text-tertiary">NULL</span> : String(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Tool results arrive as env.format_rows text: "a | b" lines, "(0 rows)", or "Error: ...". */
function ToolResult({ text }: { text: string }) {
  if (text.startsWith("Error:")) return <p className="font-mono text-body-2-regular text-destructive">{text}</p>;
  if (text === "(0 rows)") return <p className="text-body-2-regular text-text-secondary">0 rows</p>;
  const lines = text.split("\n");
  const more = lines.at(-1)!.startsWith("... (") ? lines.pop() : undefined;
  return (
    <div className="flex flex-col gap-1">
      <RowsTable rows={lines.map((l) => l.split(" | "))} />
      {more && <p className="text-body-2-regular text-text-secondary">{more.slice(4, -1)}</p>}
    </div>
  );
}

export function Turn({ segment }: { segment: Segment }) {
  if (segment.role === "tool") {
    return (
      <li className="ms-6 border-s-2 border-separator-border ps-3">
        <ToolResult text={segment.text} />
      </li>
    );
  }
  const { before, call, after } = parseTurn(segment.text);
  return (
    <li className="flex flex-col gap-2">
      {before && <p className="text-body-regular whitespace-pre-wrap text-text-primary">{before}</p>}
      {call && (
        <div className="flex flex-col gap-1">
          <span className="text-body-2-medium text-text-secondary">{call.name}</span>
          <Sql sql={call.query} />
        </div>
      )}
      {after && <p className="text-body-regular whitespace-pre-wrap text-text-primary">{after}</p>}
    </li>
  );
}

function signed(n: number) {
  return `${n > 0 ? "+" : ""}${n.toFixed(2)}`;
}

export function EpisodeRow({ episode, index }: { episode: Episode; index: number }) {
  const outcome = OUTCOMES[episode.outcome];
  return (
    <Collapsible className="rounded-2xl bg-background-secondary-default">
      <CollapsibleTrigger className="group flex w-full cursor-pointer items-center gap-3 rounded-2xl px-3 py-2 text-start outline-none focus-visible:ring-2 focus-visible:ring-border-focus-ring">
        <RiArrowRightSLine aria-hidden className="size-5 shrink-0 text-foreground-icon-secondary transition-transform group-data-[panel-open]:rotate-90" />
        <span className="w-6 shrink-0 text-body-2-regular text-text-tertiary tabular-nums">#{index + 1}</span>
        {episode.source === "teacher" && <Chip color="blue">Teacher</Chip>}
        <Chip color={outcome.chip}>{outcome.label}</Chip>
        {episode.literal && <Chip color="orange">Hard-coded answer</Chip>}
        <span className="min-w-0 flex-1 truncate font-mono text-body-2-regular text-text-secondary">{episode.final_sql}</span>
        <span className="shrink-0 text-body-2-regular text-text-secondary tabular-nums">
          {episode.turns} {episode.turns === 1 ? "turn" : "turns"} · {episode.gen_tokens} tok
          {episode.prose_tokens !== undefined && ` (${episode.prose_tokens} prose)`} · reward {episode.reward.toFixed(2)} · adv {signed(episode.advantage)}
        </span>
      </CollapsibleTrigger>
      <CollapsiblePanel>
        <div className="flex flex-col gap-4 px-4 pt-1 pb-4">
          <ol className="flex flex-col gap-3">
            {episode.segments.map((s, i) => (
              <Turn key={i} segment={s} />
            ))}
          </ol>
          {episode.verifier_reason && <p className="text-body-2-regular text-text-secondary">{episode.verifier_reason}</p>}
          {episode.answer_rows && episode.gold_rows && (
            <div className={cx("grid gap-4 rounded-xl bg-background-primary-default p-3 sm:grid-cols-2")}>
              <RowsTable rows={episode.answer_rows} caption="Graded answer" />
              <RowsTable rows={episode.gold_rows} caption="Expected" />
            </div>
          )}
        </div>
      </CollapsiblePanel>
    </Collapsible>
  );
}

/** All rollouts of one question: what GRPO compares against each other. Counts and numbering cover the whole group; `shown` filters the list. */
export function GroupCard({ episodes, shown }: { episodes: Episode[]; shown: (e: Episode) => boolean }) {
  const first = episodes[0];
  const solved = episodes.filter((e) => e.reward > 0).length;
  return (
    <section className="flex flex-col gap-3 rounded-3xl border border-separator-border p-3 sm:p-4">
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <h3 className="min-w-0 flex-1 text-headline-semibold text-text-primary">{first.question}</h3>
          <span className="shrink-0 text-body-2-medium text-text-secondary tabular-nums">
            {solved}/{episodes.length} correct · tier {first.tier} · {first.template.replaceAll("_", " ")}
          </span>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-body-2-medium text-text-secondary">Reference query</span>
          <Sql sql={first.gold_sql} />
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        {episodes.map((e, i) => shown(e) && <EpisodeRow key={i} episode={e} index={i} />)}
      </div>
    </section>
  );
}
