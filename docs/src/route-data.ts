import { getCollection } from 'astro:content';
import { defineRouteMiddleware } from '@astrojs/starlight/route-data';

import { CROWDIN_PROJECT_URL, getPathLocale, locales, splitSitePath, translatedLocales } from './config/locales.mjs';
import { getCurrentTopic } from './config/topics';
import { withBase } from './lib/base-path';
import { isLocalizablePage } from './lib/docs-page';
import {
  addCrossListings,
  fileDirectory,
  makeHrefToId,
  paginate,
  selectCrossListings,
  sortAndRelabel,
} from './lib/sidebar.mjs';

const hrefToId = makeHrefToId(import.meta.env.BASE_URL);

const getDocsIndex = async () => {
  const docs = await getCollection('docs');
  // English pages drive every locale's sidebar: hrefs map to English ids on every locale's routes (`makeHrefToId`).
  const englishDocs = docs.filter(({ id }) => !getPathLocale(id));
  return {
    /** Maps each docs page id to its file name, which keeps the sorting prefix the id drops. */
    fileNames: new Map(englishDocs.map((doc) => [doc.id, doc.filePath?.split('/').pop() ?? ''])),
    /** Maps each docs page id to its file's directory, which a custom slug or an index page's id does not show. */
    directories: new Map(englishDocs.map((doc) => [doc.id, fileDirectory(doc.filePath)])),
    crossListings: selectCrossListings(englishDocs, import.meta.env.PROD),
    /** The ids of translated pages, such as `de/users-guide/gallery/introduction`. */
    translatedIds: new Set(docs.filter(({ id }) => getPathLocale(id)).map(({ id }) => id)),
  };
};

// Registered to run before starlight-sidebar-topics (see `astro.config.mjs`), so ordering and cross-listing see every
// topic's sidebar: a page may be cross-listed in another topic. The rest runs once every other middleware is done.
export const onRequest = defineRouteMiddleware(async (context, next) => {
  const route = context.locals.starlightRoute;
  const docsIndex = await getDocsIndex();
  // Translations are Crowdin's to change: an edit to a translated file would be overwritten by its next sync. A page
  // without an edit link (`editUrl: false`) keeps none.
  if (route.locale && !route.isFallback && route.editUrl) route.editUrl = new URL(CROWDIN_PROJECT_URL);

  // Starlight lists every locale's route as a language alternate, but a fallback route is the English page (see the
  // canonical link below). Keep the alternates of the languages the page is translated into, and none without one or
  // on a fallback route, which is not the page search engines index.
  const englishId = route.locale && !route.isFallback ? route.entry.id.slice(route.locale.length + 1) : route.entry.id;
  const translatedLangs = !route.isFallback && (await isLocalizablePage(route.entry.id))
    ? translatedLocales
        .filter((locale) => docsIndex.translatedIds.has(englishId ? `${locale}/${englishId}` : locale))
        .map((locale) => locales[locale as keyof typeof locales].lang)
    : [];
  const keptLangs = new Set(translatedLangs.length > 0 ? ['x-default', locales.root.lang, ...translatedLangs] : []);
  route.head = route.head.filter(
    ({ tag, attrs }) =>
      !(tag === 'link' && attrs?.rel === 'alternate' && attrs.hreflang) || keptLangs.has(String(attrs.hreflang))
  );

  if (route.isFallback) {
    // A fallback page is the English page, so search engines should index that one in its place.
    const canonical = route.head.find(({ tag, attrs }) => tag === 'link' && attrs?.rel === 'canonical');
    const englishPath = splitSitePath(context.url.pathname, import.meta.env.BASE_URL)!.path;
    if (canonical?.attrs?.href) {
      canonical.attrs.href = new URL(withBase(englishPath, import.meta.env.BASE_URL), canonical.attrs.href).href;
    }
  }

  if (route.hasSidebar) {
    // Starlight hands each render its own copy of the sidebar.
    const { fileNames, directories, crossListings } = docsIndex;
    sortAndRelabel(route.sidebar, fileNames, hrefToId);
    addCrossListings(route.sidebar, crossListings, directories, hrefToId);
  }

  await next();

  // Starlight derived prev/next links from the whole sidebar before the reorder; keep them within the current topic,
  // whose sidebar is now the route's. This site keeps the default `pagination: true`.
  const pagination = route.hasSidebar ? paginate(route.sidebar, route.entry.data) : null;
  if (pagination) {
    route.pagination.prev = pagination.prev;
    route.pagination.next = pagination.next;
  }

  // Search scopes itself to the current topic, so only pages Pagefind indexes carry one.
  const topic = route.entry.data.pagefind !== false && getCurrentTopic(context.locals);
  if (topic) {
    route.head.push({ tag: 'meta', attrs: { name: 'docs-topic', content: topic.id, 'data-title': topic.label } });
  }
});
