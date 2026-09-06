/**
 * Root-relative paths that must never be prefixed with a locale.
 *
 * These are custom pages in `src/pages/` rather than docs-collection entries, so Starlight's
 * locale fallback never generates a `/<locale>/…` route for them and prefixing would 404.
 * Add an entry here whenever a new page lands in `src/pages/`.
 *
 * This module is the single source of truth for the two places that localize root paths:
 * the build-time `remark-localize-content` plugin and the runtime rewrite in
 * `components/MarkdownContent.astro`.
 */
export const LOCALE_EXEMPT_PATHS = ['/download'];

/** True when `path` is a root-relative docs path that should be prefixed with `locale`. */
export function shouldLocalizeRootPath(path, locale) {
  if (!path.startsWith('/') || path.startsWith('//')) {
    return false;
  }

  if (path === `/${locale}` || path.startsWith(`/${locale}/`)) {
    return false;
  }

  return !LOCALE_EXEMPT_PATHS.some((exempt) => path === exempt || path.startsWith(`${exempt}/`));
}

/** Prefixes `path` with `locale` when it points at a localizable docs route. */
export function localizeRootPath(path, locale) {
  return shouldLocalizeRootPath(path, locale) ? `/${locale}${path}` : path;
}
