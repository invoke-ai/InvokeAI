/**
 * Feedback for a canvas save to the Gallery, shared by the canvas header save and the staged-result save: the board
 * the image went to, named as the Gallery names it, in a toast fenced to the account that saved.
 */
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';
import type { Project } from '@workbench/projectContracts';
import type { TFunction } from 'i18next';

import { getGalleryBoardLabel, getGallerySettings, type GalleryBoard } from '@features/gallery/contracts';
import { galleryBoardsOptions, getGalleryListingBoardsQuery, type GalleryBoardsQuery } from '@features/gallery/queries';
import { createActionToast, toaster, type ActionToastOptions } from '@platform/ui/toaster';
import { getProjectWidgetValues } from '@workbench/widgetState';

/** Actionable toasts stay long enough to reach the button; hovering or focusing them pauses the timer. */
export const ACTION_TOAST_DURATION_MS = 8000;
/** How long a settled save waits for a boards listing it does not have before it reports without a name. */
const BOARD_LABEL_WAIT_MS = 600;

type BoardDescription = { kind: 'named'; label: string } | { kind: 'missing' } | { kind: 'unknown' };

/**
 * The board as the Gallery names it. Each listing is read from the cache at once, otherwise fetched, all within one
 * bound so feedback never waits on a slow request. Archiving a board keeps it the auto-add destination, so a board
 * absent from a listing that hides archived boards is looked for once more among them; `missing` only when a listing
 * that includes archived boards lacks it.
 */
export const describeBoard = async (
  queryClient: QueryClient,
  project: Project,
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
  const listingQuery = getGalleryListingBoardsQuery(getGallerySettings(getProjectWidgetValues(project, 'gallery')));
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

/**
 * Report to the account that saved. The toast names that account's image and board, so it leaves with the account's
 * scope rather than outlive it with buttons that would act for nobody.
 */
export const announce = (owner: AccountScope, options: ActionToastOptions): void => {
  const shown = new AbortController();
  const id = createActionToast({
    ...options,
    onStatusChange: ({ status }) => {
      if (status === 'unmounted') {
        shown.abort();
      }
    },
  });

  owner.signal.addEventListener('abort', () => toaster.dismiss(id), { once: true, signal: shown.signal });
};
