import { cx } from "@/utils/cx";

/** Placeholder brand mark. Replace with your own logo (keep it decorative: alt via the parent link's label). */
export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className={cx("size-8", className)}>
      <rect width="32" height="32" rx="9" className="fill-accent-600" />
      <path d="M9 21.5 14 10l3.2 7.2L19.5 13 23 21.5" fill="none" stroke="white" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
