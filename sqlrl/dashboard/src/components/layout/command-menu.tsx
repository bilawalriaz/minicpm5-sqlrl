import { RiFlaskLine, RiMoonLine, RiSunLine } from "@remixicon/react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { createContext, Fragment, type ReactNode, use, useEffect, useMemo, useState } from "react";
import { applyThemeWithTransition, useThemeMode } from "@/components/application/theme/theme-toggle";
import {
  Command,
  CommandCollection,
  CommandDialog,
  CommandDialogPopup,
  CommandEmpty,
  CommandFooter,
  CommandGroup,
  CommandGroupLabel,
  CommandInput,
  CommandItem,
  CommandList,
  CommandPanel,
  CommandSeparator,
} from "@/components/ui/command";
import { Kbd, KbdGroup } from "@/components/ui/kbd";
import { mainNav } from "@/config/site";
import { queries } from "@/lib/api";

/**
 * ⌘K / Ctrl+K command palette: jump to any page or run an action.
 * Pages come from `mainNav` in src/config/site.ts, plus one entry per training run.
 * Add app-specific actions to the `actions` group below.
 */

const CommandMenuContext = createContext<{ open: () => void } | null>(null);

export function useCommandMenu() {
  const ctx = use(CommandMenuContext);
  if (!ctx) throw new Error("useCommandMenu must be used inside <CommandMenuProvider>");
  return ctx;
}

interface CommandEntry {
  value: string;
  label: string;
  icon: ReactNode;
  run: () => void;
}

export function CommandMenuProvider({ children }: { children: ReactNode }) {
  const [isOpen, setOpen] = useState(false);
  const navigate = useNavigate();
  const theme = useThemeMode();
  const { data: runs } = useQuery(queries.runs());

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === "k" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        setOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const groups = useMemo(() => {
    const pages: CommandEntry[] = [
      ...mainNav.map((item) => ({
        value: `page:${item.to}`,
        label: item.label,
        icon: <item.icon aria-hidden />,
        run: () => void navigate({ to: item.to }),
      })),
      ...(runs ?? []).map((run) => ({
        value: `run:${run.name}`,
        label: run.name,
        icon: <RiFlaskLine aria-hidden />,
        run: () => void navigate({ to: "/runs/$name", params: { name: run.name } }),
      })),
    ];
    const actions: CommandEntry[] = [
      {
        value: "action:theme",
        label: theme === "dark" ? "Switch to light theme" : "Switch to dark theme",
        icon: theme === "dark" ? <RiSunLine aria-hidden /> : <RiMoonLine aria-hidden />,
        run: () => void applyThemeWithTransition(theme === "dark" ? "light" : "dark"),
      },
    ];
    return [
      { value: "Pages", items: pages },
      { value: "Actions", items: actions },
    ];
  }, [navigate, theme, runs]);

  const ctx = useMemo(() => ({ open: () => setOpen(true) }), []);

  return (
    <CommandMenuContext value={ctx}>
      {children}
      <CommandDialog open={isOpen} onOpenChange={setOpen}>
        <CommandDialogPopup aria-label="Command menu">
          <Command
            items={groups}
            itemToStringValue={(item: unknown) => (item as CommandEntry).label}
          >
            <CommandInput placeholder="Search pages and actions…" />
            <CommandPanel>
              <CommandEmpty>No results found.</CommandEmpty>
              <CommandList>
                {(group: (typeof groups)[number], index: number) => (
                  <Fragment key={group.value}>
                    <CommandGroup items={group.items}>
                      <CommandGroupLabel>{group.value}</CommandGroupLabel>
                      <CommandCollection>
                        {(item: CommandEntry) => (
                          <CommandItem
                            key={item.value}
                            value={item}
                            onClick={() => {
                              setOpen(false);
                              item.run();
                            }}
                          >
                            {item.icon}
                            {item.label}
                          </CommandItem>
                        )}
                      </CommandCollection>
                    </CommandGroup>
                    {index < groups.length - 1 && <CommandSeparator />}
                  </Fragment>
                )}
              </CommandList>
            </CommandPanel>
            <CommandFooter>
              <span className="flex items-center gap-2">
                <KbdGroup>
                  <Kbd>↑</Kbd>
                  <Kbd>↓</Kbd>
                </KbdGroup>
                Navigate
              </span>
              <span className="flex items-center gap-2">
                <Kbd>↵</Kbd>
                Open
              </span>
            </CommandFooter>
          </Command>
        </CommandDialogPopup>
      </CommandDialog>
    </CommandMenuContext>
  );
}
