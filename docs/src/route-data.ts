import { getCollection } from 'astro:content';
import { defineRouteMiddleware } from '@astrojs/starlight/route-data';

import { getCurrentTopic } from './config/topics';
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
  return {
    /** Maps each docs page id to its file name, which keeps the sorting prefix the id drops. */
    fileNames: new Map(docs.map((doc) => [doc.id, doc.filePath?.split('/').pop() ?? ''])),
    /** Maps each docs page id to its file's directory, which a custom slug or an index page's id does not show. */
    directories: new Map(docs.map((doc) => [doc.id, fileDirectory(doc.filePath)])),
    crossListings: selectCrossListings(docs, import.meta.env.PROD),
  };
};

// Registered to run before starlight-sidebar-topics (see `astro.config.mjs`), so ordering and cross-listing see every
// topic's sidebar: a page may be cross-listed in another topic. The rest runs once every other middleware is done.
export const onRequest = defineRouteMiddleware(async (context, next) => {
  const route = context.locals.starlightRoute;
  if (route.hasSidebar) {
    // Starlight hands each render its own copy of the sidebar.
    const { fileNames, directories, crossListings } = await getDocsIndex();
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
