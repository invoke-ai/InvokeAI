/**
 * Feedback for a canvas save to the Gallery, shared by the canvas header save and the staged-result save: the board
 * the image went to, named as the Gallery names it, in a toast fenced to the account that saved.
 */
import type { QueryClient } from '@tanstack/react-query';
import type { Project } from '@workbench/projectContracts';
import type { ScopedDefaults } from '@workbench/settings/contracts';
import type { TFunction } from 'i18next';

import { getGalleryBoardLabel, getGallerySettings, type GalleryBoard } from '@features/gallery/contracts';
import { galleryBoardsOptions, getGalleryListingBoardsQuery, type GalleryBoardsQuery } from '@features/gallery/queries';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';

/** How long a settled save waits for a boards listing it does not have before it reports without a name. */
const BOARD_LABEL_WAIT_MS = 600;

type BoardDescription = { kind: 'named'; label: string } | { kind: 'missing' } | { kind: 'unknown' };

/**
 * The board as the Gallery names it. Each listing is read from the cache at once, otherwise fetched, all within one
 * bound so feedback never waits on a slow request. Archiving a board keeps it the auto-add destination, so a board
 * absent from a listing that hides archived boards is looked for once more among them; `missing` only when a listing
 * that includes archived boards lacks it. The listing is the one the Gallery shows, read through the user's defaults so
 * its query key matches the live Gallery's.
 */
export const describeBoard = async (
  queryClient: QueryClient,
  project: Project,
  defaults: ScopedDefaults,
  boardId: string,
  t: TFunction
): Promise<BoardDescription> => {
  if (boardId === 'none') {
    return { kind: 'named', label: t('widgets.gallery.uncategorized') };
  }

  const deadline = new Promise<null>((resolve) => {
    globalThis.setTimeout(() => resolve(null), BOARD_LABEL_WAIT_MS);
  });
  const readBoards = (query: GalleryBoardsQuery): Promise<GalleryBoard[] | null> => {
    const options = galleryBoardsOptions(query);
    const cached = queryClient.getQueryData<GalleryBoard[]>(options.queryKey);

    return cached
      ? Promise.resolve(cached)
      : Promise.race([queryClient.fetchQuery(options).catch(() => null), deadline]);
  };
  const describe = (boards: GalleryBoard[]): BoardDescription | null => {
    const board = boards.find((candidate) => candidate.id === boardId);

    return board ? { kind: 'named', label: getGalleryBoardLabel(board, t) } : null;
  };
  const listingQuery = getGalleryListingBoardsQuery(
    getGallerySettings(getResolvedProjectWidgetValues(project, 'gallery', defaults))
  );
  const listing = await readBoards(listingQuery);
  const listed = listing && describe(listing);

  if (listed) {
    return listed;
  }

  const everything = listingQuery.includeArchived
    ? listing
    : await readBoards({ ...listingQuery, includeArchived: true });

  if (!everything) {
    return { kind: 'unknown' };
  }

  return describe(everything) ?? { kind: 'missing' };
};
