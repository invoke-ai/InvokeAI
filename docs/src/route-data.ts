import { getCollection } from 'astro:content';
import { defineRouteMiddleware } from '@astrojs/starlight/route-data';

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

export const onRequest = defineRouteMiddleware(async (context) => {
  const route = context.locals.starlightRoute;
  // Starlight hands each render its own copy of the sidebar.
  const { sidebar } = route;
  const { fileNames, directories, crossListings } = await getDocsIndex();
  const reordered = sortAndRelabel(sidebar, fileNames, hrefToId);
  const crossListed = addCrossListings(sidebar, crossListings, directories, hrefToId);
  if (!reordered && !crossListed) return;

  // Starlight derived prev/next links before the reorder. This site keeps the default `pagination: true`.
  const pagination = paginate(sidebar, route.entry.data);
  if (!pagination) return;
  route.pagination.prev = pagination.prev;
  route.pagination.next = pagination.next;
});
