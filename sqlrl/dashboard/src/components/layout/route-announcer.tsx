import { useRouterState } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";

/**
 * Accessibility for client-side navigation. A SPA route change does not
 * trigger a page load, so screen readers hear nothing and keyboard focus stays
 * on the link that was clicked. After each navigation this:
 *   1. announces the new document title in a polite live region, and
 *   2. moves focus to the page's <main id="main"> landmark.
 */
export function RouteAnnouncer() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isLoading = useRouterState({ select: (s) => s.status === "pending" });
  const [message, setMessage] = useState("");
  const firstRender = useRef(true);

  useEffect(() => {
    if (isLoading) return;
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    // Wait a frame so React 19 has hoisted the new page's <title>.
    const frame = requestAnimationFrame(() => {
      setMessage(document.title);
      document.getElementById("main")?.focus({ preventScroll: true });
    });
    return () => cancelAnimationFrame(frame);
  }, [pathname, isLoading]);

  return (
    <div aria-live="polite" aria-atomic="true" className="sr-only">
      {message}
    </div>
  );
}
