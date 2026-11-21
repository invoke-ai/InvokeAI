// The docs locales. English is the root locale, served without a prefix; each translated locale is served under
// `/<locale>/`. Crowdin adds translated pages to `src/content/docs/<locale>/`, mirroring the English tree, and
// Starlight serves the English page wherever a translation is missing. Plain JavaScript so plugins, scripts and tests
// can import it.

/** Starlight's `locales` option. Keys are lowercase, as the URL segments they become. */
export const locales = {
  root: { label: 'English', lang: 'en' },
  de: { label: 'Deutsch', lang: 'de' },
  es: { label: 'Español', lang: 'es' },
  hi: { label: 'हिन्दी', lang: 'hi' },
};

export const translatedLocales = Object.keys(locales).filter((locale) => locale !== 'root');

/**
 * Paths, without the base path, that only exist at the site root: pages that are not docs and so have no locale
 * routes (releases are English GitHub content, see `astro.config.mjs`), and build output. Links to them never take a
 * locale prefix.
 */
export const ROOT_ONLY_PATHS = [
  '/download',
  '/showcase',
  '/releases',
  '/_astro',
  '/pagefind',
  '/settings-search.json',
  '/app-search.json',
];

/** Translators work in the Crowdin project, which opens pull requests with their translations. */
export const CROWDIN_PROJECT_URL = 'https://crowdin.com/project/invoke';

/**
 * The translated locale a path without the base path, or a docs entry id, starts with; `undefined` for the root
 * locale.
 * @param {string} path
 * @returns {string | undefined}
 */
export const getPathLocale = (path) => {
  const segment = path.replace(/^\/+/, '').split(/[/?#]/)[0];
  return translatedLocales.includes(segment) ? segment : undefined;
};

/**
 * A site URL path as its locale and the English page's path, both without the base path: `/InvokeAI/de/x/#y` with base
 * `/InvokeAI` is `{ locale: 'de', path: '/x/#y' }`. `undefined` for a path outside the base path.
 * @param {string} pathname a root-relative path, which may end in a query or fragment
 * @param {string} base Astro's `BASE_URL`, such as `/` or `/InvokeAI`
 * @returns {{ locale: string | undefined, path: string } | undefined}
 */
export const splitSitePath = (pathname, base) => {
  const basePath = base.replace(/\/+$/, '');
  const rest = pathname.slice(basePath.length);
  if (!pathname.startsWith(basePath) || !/^(?:$|[/?#])/.test(rest)) return undefined;
  const path = rest.startsWith('/') ? rest : `/${rest}`;
  const locale = getPathLocale(path);
  if (!locale) return { locale, path };
  const englishPath = path.slice(locale.length + 1);
  return { locale, path: englishPath.startsWith('/') ? englishPath : `/${englishPath}` };
};

/** @param {string} path a path without the base path, which may end in a query or fragment */
export const isRootOnlyPath = (path) => {
  const pathname = path.split(/[?#]/)[0];
  return ROOT_ONLY_PATHS.some((rootPath) => pathname === rootPath || pathname.startsWith(`${rootPath}/`));
};

/**
 * A root-relative path without the base path, moved into `locale`. Paths that already carry a locale, and root-only
 * paths, stay as they are.
 * @param {string} path
 * @param {string | undefined} locale a translated locale, or `undefined` for the root locale
 */
export const localizePath = (path, locale) =>
  !locale || getPathLocale(path) || isRootOnlyPath(path) ? path : `/${locale}${path}`;
