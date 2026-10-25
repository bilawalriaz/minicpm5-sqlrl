import { RiCloseLine, RiSearchLine, RiSideBarFill } from "@remixicon/react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { type ReactNode, useEffect, useState } from "react";
import { ThemeToggle } from "@/components/application/theme/theme-toggle";
import { Kbd } from "@/components/base/kbd/kbd";
import { useCommandMenu } from "@/components/layout/command-menu";
import { Logo } from "@/components/layout/logo";
import { mainNav, type NavItem, site } from "@/config/site";
import { queries } from "@/lib/api";
import { phaseLabel } from "@/features/runs/status";
import { cx } from "@/utils/cx";

/**
 * Floating sidebar, adapted from BoardUI's dashboard sidebar block: same
 * panel, rail-collapse animation and selected state, but driven by
 * src/config/site.ts, wired to the router, and showing the real user.
 */

const COLLAPSED_KEY = "app:sidebar-collapsed";

function readCollapsed() {
  try {
    return localStorage.getItem(COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

/** Label slot that blurs and shrinks away when the rail collapses. */
function Collapsible({ collapsed, children }: { collapsed: boolean; children: ReactNode }) {
  return (
    <span
      aria-hidden={collapsed || undefined}
      className={cx(
        "flex min-w-0 items-center overflow-hidden transition-[max-width,opacity,filter] duration-300 ease-in-out",
        collapsed ? "max-w-0 opacity-0 blur-[3px]" : "max-w-full opacity-100 blur-0",
      )}
    >
      {children}
    </span>
  );
}

function NavLink({ item, collapsed, onNavigate }: { item: NavItem; collapsed: boolean; onNavigate?: () => void }) {
  return (
    <Link
      to={item.to}
      onClick={onNavigate}
      activeOptions={{ exact: item.to === "/" }}
      aria-label={collapsed ? item.label : undefined}
      title={collapsed ? item.label : undefined}
      className={cx(
        "group flex items-center overflow-hidden rounded-2lg p-2 outline-none",
        "transition-[width,background-color] duration-300 ease-in-out",
        "focus-visible:ring-2 focus-visible:ring-border-focus-ring",
        collapsed ? "w-9" : "w-full",
        "hover:bg-background-secondary-hover",
        "data-[status=active]:bg-linear-to-b data-[status=active]:from-accent-500 data-[status=active]:to-accent-600 data-[status=active]:shadow-nav-selected",
      )}
    >
      <span className="flex min-w-0 items-center gap-2">
        <item.icon
          aria-hidden
          className="size-5 shrink-0 text-foreground-icon-secondary group-data-[status=active]:text-white"
        />
        <Collapsible collapsed={collapsed}>
          <span className="text-body-medium whitespace-nowrap text-text-secondary group-data-[status=active]:text-white">
            {item.label}
          </span>
        </Collapsible>
      </span>
    </Link>
  );
}

/** Every run, newest first, so switching runs is one click from anywhere. */
function RunLinks({ onNavigate }: { onNavigate?: () => void }) {
  const { data: runs } = useQuery(queries.runs());
  if (!runs?.length) return null;
  return (
    <nav aria-label="Runs" className="flex w-full flex-col gap-0.5 px-0.5">
      {runs.map((run) => (
        <Link
          key={run.name}
          to="/runs/$name"
          params={{ name: run.name }}
          onClick={onNavigate}
          className={cx(
            "flex flex-col rounded-2lg px-2 py-1.5 outline-none hover:bg-background-secondary-hover",
            "focus-visible:ring-2 focus-visible:ring-border-focus-ring data-[status=active]:bg-background-tertiary-default",
          )}
        >
          <span className="truncate text-body-medium text-text-primary">{run.name}</span>
          <span className="truncate text-body-2-regular text-text-secondary">{phaseLabel(run.status)}</span>
        </Link>
      ))}
    </nav>
  );
}

export function AppSidebar({
  mobile = false,
  onClose,
  className,
}: {
  /** Rendered inside the mobile drawer: always expanded, close button instead of collapse. */
  mobile?: boolean;
  onClose?: () => void;
  className?: string;
}) {
  const [collapsedState, setCollapsed] = useState(readCollapsed);
  const collapsed = mobile ? false : collapsedState;
  const commandMenu = useCommandMenu();

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSED_KEY, collapsedState ? "1" : "0");
    } catch {
      // ignore
    }
  }, [collapsedState]);

  return (
    <aside
      aria-label="Sidebar"
      className={cx(
        "flex h-full shrink-0 flex-col justify-between gap-3 overflow-hidden",
        mobile
          ? "w-full p-3"
          : "rounded-3xl border border-border-button-white bg-background-secondary-default shadow-sidebar",
        "transition-[width] duration-300 ease-in-out",
        !mobile && (collapsed ? "w-[60px] px-[11px] py-3" : "w-[260px] p-3"),
        className,
      )}
    >
      <div className="-m-2 flex min-h-0 w-[calc(100%+16px)] flex-col gap-3 overflow-y-auto p-2 [scrollbar-width:none]">
        {/* Brand + collapse */}
        <div
          className={cx(
            "flex w-full transition-[gap] duration-300 ease-in-out",
            collapsed ? "flex-col-reverse items-start gap-2.5" : "flex-row items-center justify-between",
          )}
        >
          <Link
            to="/"
            onClick={onClose}
            className="flex min-w-0 items-center gap-2 rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-border-focus-ring"
            aria-label={`${site.name} home`}
          >
            <Logo className={cx("size-9 shrink-0", collapsed && "size-9")} />
            <Collapsible collapsed={collapsed}>
              <span className="text-headline-semibold whitespace-nowrap text-text-primary">{site.name}</span>
            </Collapsible>
          </Link>
          {mobile ? (
            <button
              type="button"
              aria-label="Close navigation"
              onClick={onClose}
              className="cursor-pointer rounded-md p-1 text-foreground-icon-secondary outline-none focus-visible:ring-2 focus-visible:ring-border-focus-ring"
            >
              <RiCloseLine className="size-5" aria-hidden />
            </button>
          ) : (
            <button
              type="button"
              aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
              aria-expanded={!collapsed}
              onClick={() => setCollapsed((v) => !v)}
              className={cx(
                "cursor-pointer rounded-md text-foreground-icon-secondary outline-none focus-visible:ring-2 focus-visible:ring-border-focus-ring",
                collapsed && "flex w-9 items-center justify-center",
              )}
            >
              <RiSideBarFill
                aria-hidden
                className={cx("size-5 transition-transform duration-300", collapsed ? "rtl:-scale-x-100" : "-scale-x-100 rtl:scale-x-100")}
              />
            </button>
          )}
        </div>

        {/* Search → command palette */}
        <button
          type="button"
          onClick={() => {
            onClose?.();
            commandMenu.open();
          }}
          aria-label={collapsed ? "Search" : undefined}
          aria-keyshortcuts="Control+K Meta+K"
          title={collapsed ? "Search" : undefined}
          className={cx(
            "flex cursor-pointer items-center gap-2 rounded-full bg-background-tertiary-default p-2 outline-none",
            "transition-[width,background-color] duration-300 ease-in-out hover:bg-background-tertiary-hover",
            "focus-visible:ring-2 focus-visible:ring-border-focus-ring",
            collapsed ? "w-9" : "w-full",
          )}
        >
          <span className={cx("flex min-w-0 items-center gap-2", !collapsed && "flex-1")}>
            <RiSearchLine className="size-5 shrink-0 text-foreground-icon-secondary" aria-hidden />
            <Collapsible collapsed={collapsed}>
              <span className="text-body-medium whitespace-nowrap text-text-secondary">Search</span>
            </Collapsible>
          </span>
          <Collapsible collapsed={collapsed}>
            <Kbd>⌘K</Kbd>
          </Collapsible>
        </button>

        <nav aria-label="Main" className={cx("flex w-full flex-col gap-1", !collapsed && "px-0.5")}>
          {mainNav.map((item) => (
            <NavLink key={item.to} item={item} collapsed={collapsed} onNavigate={onClose} />
          ))}
        </nav>
        {!collapsed && <RunLinks onNavigate={onClose} />}
      </div>

      <div className="flex w-full shrink-0 flex-col gap-3">
        {collapsed ? <ThemeToggle collapsed /> : <ThemeToggle appearance="sidebar-segmented" />}
      </div>
    </aside>
  );
}
