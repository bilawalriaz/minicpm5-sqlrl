import { RiFlaskLine } from "@remixicon/react";
import type { ComponentType } from "react";

export const site = {
  name: import.meta.env.VITE_APP_NAME ?? "SQL RL",
  description: "Live view of GRPO training runs for the SQLite agent.",
  /** Shown in the browser tab after the page title: "run_1 · SQL RL". */
  titleSeparator: " · ",
} as const;

type IconComponent = ComponentType<{ className?: string; "aria-hidden"?: boolean | "true" | "false" }>;

export interface NavItem {
  /** Route path; must exist in src/routes. */
  to: "/";
  label: string;
  icon: IconComponent;
}

/** Drives the sidebar, the mobile drawer and the command palette. */
export const mainNav: NavItem[] = [{ to: "/", label: "Runs", icon: RiFlaskLine }];

export function pageTitle(title?: string) {
  return title ? `${title}${site.titleSeparator}${site.name}` : site.name;
}
