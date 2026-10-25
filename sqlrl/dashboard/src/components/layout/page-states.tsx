import { RiCompass3Line, RiErrorWarningLine } from "@remixicon/react";
import { type ErrorComponentProps, Link, useRouter } from "@tanstack/react-router";
import { PageMeta } from "@/components/layout/page-meta";
import { Button } from "@/components/ui/button";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";

/** Shown while a route's loader is running for longer than `defaultPendingMs`. */
export function PendingPage() {
  return (
    <div className="flex flex-col gap-6" aria-busy="true">
      <span className="sr-only" role="status">
        Loading
      </span>
      <div className="flex flex-col gap-2">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-4 w-72" />
      </div>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="h-28 rounded-2xl" />
        ))}
      </div>
      <Skeleton className="h-80 rounded-2xl" />
    </div>
  );
}

/** Full-screen variant used before the app shell exists (auth check). */
export function FullPageSpinner() {
  return (
    <div className="grid min-h-dvh place-items-center" role="status" aria-label="Loading">
      <Spinner />
    </div>
  );
}

export function NotFoundPage() {
  return (
    <main id="main" tabIndex={-1} className="grid min-h-[60dvh] place-items-center p-6 outline-none">
      <PageMeta title="Page not found" />
      <Empty>
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <RiCompass3Line aria-hidden />
          </EmptyMedia>
          <EmptyTitle>
            <h1>Page not found</h1>
          </EmptyTitle>
          <EmptyDescription>The page you’re looking for doesn’t exist or has moved.</EmptyDescription>
        </EmptyHeader>
        <EmptyContent>
          <Button render={<Link to="/" />}>Go to overview</Button>
        </EmptyContent>
      </Empty>
    </main>
  );
}

export function ErrorPage({ error, reset }: ErrorComponentProps) {
  const router = useRouter();
  return (
    <div role="alert" className="grid min-h-[60dvh] place-items-center p-6">
      <PageMeta title="Something went wrong" />
      <Empty>
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <RiErrorWarningLine aria-hidden />
          </EmptyMedia>
          <EmptyTitle>
            <h1>Something went wrong</h1>
          </EmptyTitle>
          <EmptyDescription>{error instanceof Error ? error.message : "An unexpected error occurred."}</EmptyDescription>
        </EmptyHeader>
        <EmptyContent>
          <Button
            onClick={() => {
              reset();
              void router.invalidate();
            }}
          >
            Try again
          </Button>
        </EmptyContent>
      </Empty>
    </div>
  );
}
