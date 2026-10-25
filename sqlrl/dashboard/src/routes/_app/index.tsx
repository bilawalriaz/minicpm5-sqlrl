import { useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { PageHeader } from "@/components/layout/page-header";
import { PageMeta } from "@/components/layout/page-meta";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { phaseLabel } from "@/features/runs/status";
import { queries } from "@/lib/api";
import { format } from "@/lib/format";

export const Route = createFileRoute("/_app/")({
  loader: ({ context }) => context.queryClient.ensureQueryData(queries.runs()),
  component: RunsPage,
});

const score = (n: number | null) => (n === null ? "" : n.toFixed(3));

function RunsPage() {
  const { data: runs } = useSuspenseQuery(queries.runs());

  return (
    <div className="flex flex-col gap-6">
      <PageMeta title="Runs" />
      <PageHeader title="Runs" />
      {runs.length === 0 ? (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>No runs yet</EmptyTitle>
            <EmptyDescription>
              Start one with <code className="font-mono">python train.py</code> in the sqlrl folder.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <Table variant="card">
          <TableHeader>
            <TableRow>
              <TableHead>Run</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-end">Step</TableHead>
              <TableHead className="text-end">Reward</TableHead>
              <TableHead className="text-end">Eval pass@1</TableHead>
              <TableHead>Started</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {runs.map((run) => (
              <TableRow key={run.name}>
                <TableCell>
                  <Link to="/runs/$name" params={{ name: run.name }} className="text-body-medium text-text-primary underline-offset-4 hover:underline">
                    {run.name}
                  </Link>
                </TableCell>
                <TableCell className="text-text-secondary">{phaseLabel(run.status)}</TableCell>
                <TableCell className="text-end tabular-nums">
                  {run.step} / {run.steps}
                </TableCell>
                <TableCell className="text-end tabular-nums">{score(run.reward)}</TableCell>
                <TableCell className="text-end tabular-nums">
                  {run.evalFirst === null ? "" : run.evalLast === run.evalFirst ? score(run.evalFirst) : `${score(run.evalFirst)} → ${score(run.evalLast)}`}
                </TableCell>
                <TableCell className="text-text-secondary">{format.relative(new Date(run.started * 1000).toISOString())}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
