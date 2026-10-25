import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";
import { ErrorPage, NotFoundPage } from "@/components/layout/page-states";
import { RouteAnnouncer } from "@/components/layout/route-announcer";
import { ToastProvider } from "@/components/ui/toast";

export interface RouterContext {
  queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
  notFoundComponent: NotFoundPage,
  errorComponent: ErrorPage,
});

function RootLayout() {
  return (
    <ToastProvider>
      <Outlet />
      <RouteAnnouncer />
    </ToastProvider>
  );
}
