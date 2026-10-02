import type { GalleryView } from '@features/gallery';
import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryItemsFilter } from '@features/gallery/queries';
import type { QueryClient } from '@tanstack/react-query';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';

import { galleryItems, toGalleryItemKey } from '@features/gallery';
import { getGallerySettings, isGalleryNavigationCurrent, requestGalleryItemReveal } from '@features/gallery/contracts';
import {
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  galleryBoardsOptions,
  galleryItemNamesOptions,
  galleryItemsInfiniteOptions,
} from '@features/gallery/queries';
import { getProjectWidgetValues } from '@workbench/widgetState';

/** Inject workbench dependencies so reveal loads on demand without adding gallery transfer code to editor boot. */
export interface GalleryRevealContext {
  commands: WorkbenchCommands;
  queries: WorkbenchQueries;
  queryClient: QueryClient;
}

/**
 * What the gesture was, at the moment it was made. Both fields are claimed by
 * the CALLER and neither may be re-read here: a caller that loads this module
 * on demand would otherwise take its ordering, and its project, from whenever
 * the chunk happened to land.
 */
export interface GalleryRevealTicket {
  /** The project the press belongs to; its writes may not land in another. */
  projectId: string;
  /** This navigation's place in the global ordering; see `claimGalleryNavigationSequence`. */
  sequence: number;
}

/**
 * Extends the infinite window until it covers `pagesNeeded` pages. This must
 * NOT be a plain prefetch: the mounted gallery keeps the query fresh, and
 * `fetchQuery` returns fresh cache without honoring the `pages` option — the
 * reveal has to force the fetch (staleTime 0) or the window never grows. Two
 * passes because a concurrent fetch already in flight (a second rapid click)
 * absorbs the call without extending; the retry runs after it settles.
 */
const ensureGalleryPagesLoaded = async (
  queryClient: QueryClient,
  listingFilter: GalleryItemsFilter,
  pagesNeeded: number
): Promise<void> => {
  const options = galleryItemsInfiniteOptions(listingFilter, { kind: 'infinite' });

  for (let attempt = 0; attempt < 2; attempt += 1) {
    const data = queryClient.getQueryData<{ pages: unknown[] }>(options.queryKey);

    if ((data?.pages.length ?? 0) >= pagesNeeded) {
      return;
    }

    await queryClient.fetchInfiniteQuery({ ...options, pages: pagesNeeded, staleTime: 0 });
  }
};

/**
 * Reveal a freshly resolved item in its board/view with filters cleared and its page loaded; selection drives grid
 * scrolling and Preview. Do not raise widgets. Caller-minted gesture tickets fence project changes and later
 * selections even across lazy loading. Hydration failures reject without changing selection; position failures
 * only lose scrolling.
 */
export const revealGalleryItem = (
  { commands, queries, queryClient }: GalleryRevealContext,
  ref: GalleryItemRef,
  { projectId, sequence }: GalleryRevealTicket
): Promise<void> => {
  // Fence the network result to the gesture's project before clearing filters or selecting.
  const isCurrent = () => isGalleryNavigationCurrent(sequence) && queries.isActiveProject(projectId);

  return galleryItems.resolve(ref).then(async (image) => {
    if (!isCurrent()) {
      return;
    }

    const getGalleryValues = () => getProjectWidgetValues(queries.getSnapshot().activeProject, 'gallery');
    const settings = getGallerySettings(getGalleryValues());
    const targetView: GalleryView = image.category === 'general' ? 'images' : 'assets';
    // Match the post-reveal listing filters and starred partition so prefetched pages enter the gallery's cache.
    const wantsStarredOnly = image.starred === true;
    const listingFilter = {
      boardId: image.boardId,
      galleryView: targetView,
      orderDir: settings.imageOrderDir,
      searchTerm: '',
      starred: wantsStarredOnly,
    };
    // Resolve position for paging, but preserve selection on failure. Check board visibility to avoid using a
    // hidden board's page in Uncategorized.
    let boardIndex: number | null = null;

    try {
      const boardsPromise = queryClient
        .fetchQuery(
          galleryBoardsOptions({
            includeArchived: settings.showArchivedBoards,
            includeDateBoards: settings.showDateBoards,
            orderBy: settings.boardOrderBy,
            orderDir: settings.boardOrderDir,
          })
        )
        // Unknown beats blocked: without the boards list the reveal
        // proceeds as if the board were listable.
        .catch(() => null);
      const names = await queryClient.fetchQuery(galleryItemNamesOptions(listingFilter));
      const boards = await boardsPromise;
      const index = names.items.findIndex((item) => item.kind === ref.kind && item.name === ref.name);
      const isBoardListable =
        image.boardId === 'none' ||
        boards === null ||
        boards.length === 0 ||
        boards.some((board) => board.id === image.boardId);

      boardIndex = index >= 0 && isBoardListable ? index : null;
    } catch {
      boardIndex = null;
    }

    if (!isCurrent()) {
      return;
    }

    const values = getGalleryValues();
    const settingsNow = getGallerySettings(values);

    // The listing's ordering may have changed while the name list was
    // in flight (sort direction); the computed index describes the old
    // ordering, so the page landing is dropped.
    if (settingsNow.imageOrderDir !== settings.imageOrderDir) {
      boardIndex = null;
    }

    const currentView: GalleryView = values.galleryView === 'assets' ? 'assets' : 'images';
    const hasSearch =
      (typeof values.searchTerm === 'string' && values.searchTerm !== '') ||
      typeof values.semanticSearchText === 'string';

    // Clear filters to match the indexed listing and select the item's starred partition.
    if (
      hasSearch ||
      (values.starredOnly === true) !== wantsStarredOnly ||
      (values.semanticImageQuery !== null && values.semanticImageQuery !== undefined)
    ) {
      commands.widgets.patchValues('gallery', {
        searchTerm: '',
        semanticImageQuery: null,
        semanticSearchText: null,
        starredOnly: wantsStarredOnly,
      });
    }

    if (currentView !== targetView) {
      commands.gallery.setView(targetView);
    }

    // Select the board before the item so its navigation query captures the destination listing for Preview
    // next/previous.
    commands.gallery.selectBoard(image.boardId);

    const page = boardIndex !== null ? Math.floor(boardIndex / GALLERY_PAGE_SIZE) : null;

    if (page !== null && settingsNow.paginationMode === 'paginated') {
      commands.gallery.setPage(page);
    }

    if (boardIndex !== null && page !== null && settingsNow.paginationMode === 'infinite') {
      if (boardIndex < GALLERY_MAX_ROWS) {
        // Load pages through the item without delaying selection; the grid completes its reveal when the item
        // arrives.
        void ensureGalleryPagesLoaded(queryClient, listingFilter, page + 1).catch(() => {});
      } else {
        // Deeper than the base window can ever load: anchor the
        // infinite window at the image's page instead (the mounted
        // gallery query fetches it on its own). Any board, search, or
        // view change resets the anchor back to the top.
        commands.gallery.setPage(page);
      }
    }

    commands.gallery.selectItem(image, projectId, page ?? undefined);
    requestGalleryItemReveal(toGalleryItemKey(ref));
  });
};
