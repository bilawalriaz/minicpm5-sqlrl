import { useLocation } from "@tanstack/react-router";
import { env } from "@/config/env";
import { pageTitle, site } from "@/config/site";

/**
 * Per-page <head> tags. React 19 hoists <title>, <meta> and <link> rendered
 * anywhere in the tree into <head>, so no helmet library is needed.
 *
 * Authenticated pages default to `noindex`: dashboards should not appear in
 * search results. Public pages (login, marketing) pass `index`.
 */
export function PageMeta({
  title,
  description = site.description,
  index = false,
}: {
  title?: string;
  description?: string;
  index?: boolean;
}) {
  const { pathname } = useLocation();
  const fullTitle = pageTitle(title);
  const canonical = env.siteUrl ? `${env.siteUrl}${pathname}` : undefined;

  return (
    <>
      <title>{fullTitle}</title>
      <meta name="description" content={description} />
      <meta name="robots" content={index ? "index, follow" : "noindex, nofollow"} />
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={description} />
      {canonical && <link rel="canonical" href={canonical} />}
      {canonical && <meta property="og:url" content={canonical} />}
    </>
  );
}
