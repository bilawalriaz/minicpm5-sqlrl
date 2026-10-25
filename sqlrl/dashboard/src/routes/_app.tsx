import { RiMenuLine, RiSearchLine } from "@remixicon/react";
import { createFileRoute, Outlet } from "@tanstack/react-router";
import { useState } from "react";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { CommandMenuProvider, useCommandMenu } from "@/components/layout/command-menu";
import { Button } from "@/components/ui/button";
import { Sheet, SheetPopup, SheetTitle } from "@/components/ui/sheet";

/** App shell. Every route file under src/routes/_app/ renders inside it; `_app` does not appear in URLs. */
export const Route = createFileRoute("/_app")({
  component: AppLayout,
});

function AppLayout() {
  return (
    <CommandMenuProvider>
      <Shell />
    </CommandMenuProvider>
  );
}

function Shell() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const commandMenu = useCommandMenu();

  return (
    <div className="flex min-h-dvh bg-background-full">
      <a
        href="#main"
        className="sr-only z-50 rounded-md bg-background-primary-default px-3 py-2 text-body-medium text-text-primary shadow-dropdown focus:not-sr-only focus:fixed focus:top-3 focus:left-3"
      >
        Skip to content
      </a>

      {/* Desktop sidebar */}
      <div className="sticky top-0 hidden h-dvh p-3 lg:block">
        <AppSidebar />
      </div>

      {/* Mobile drawer */}
      <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
        <SheetPopup side="left" showCloseButton={false} className="w-[min(300px,85vw)] p-0">
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          <AppSidebar mobile onClose={() => setMobileNavOpen(false)} />
        </SheetPopup>
      </Sheet>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-2 border-b border-separator-border bg-background-full/85 px-4 backdrop-blur-md lg:hidden">
          <Button
            variant="ghost"
            size="icon"
            aria-label="Open navigation"
            aria-expanded={mobileNavOpen}
            onClick={() => setMobileNavOpen(true)}
          >
            <RiMenuLine aria-hidden />
          </Button>
          <div className="flex-1" />
          <Button variant="ghost" size="icon" aria-label="Search" onClick={commandMenu.open}>
            <RiSearchLine aria-hidden />
          </Button>
        </header>

        <main id="main" tabIndex={-1} className="mx-auto w-full max-w-7xl flex-1 px-4 pt-4 pb-10 outline-none sm:px-6 lg:pt-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
