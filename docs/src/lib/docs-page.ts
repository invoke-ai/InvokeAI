import { getCollection } from 'astro:content';

import { isRootOnlyPath } from '../config/locales.mjs';

/**
 * Whether a route renders a docs collection page. Its entry id is the page's own for a translation, and the English
 * page's for a locale's fallback route. Other routes, such as the 404 page, release pages and hand-written
 * `StarlightPage`s, carry entries outside the collection.
 */
export const isDocsPage = async (entryId: string) => {
  // Starlight's route entries give the home page the id `''`, which is `index` in the collection.
  const id = entryId || 'index';
  // `getEntry` would log a warning for every page outside the collection.
  return (await getCollection('docs', (doc) => doc.id === id)).length > 0;
};

/** Whether a route exists in every locale, so it can link to its translations: docs pages that are not root-only. */
export const isLocalizablePage = async (entryId: string) =>
  !isRootOnlyPath(`/${entryId}`) && (await isDocsPage(entryId));
