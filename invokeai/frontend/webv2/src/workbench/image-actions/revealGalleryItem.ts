import type { GalleryView } from '@features/gallery';
import type { GalleryItemRef } from '@features/gallery/contracts';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';

import { galleryItems, toGalleryItemKey } from '@features/gallery';
import { getGallerySettings, isGalleryNavigationCurrent, requestGalleryItemReveal } from '@features/gallery/contracts';
import { fetchVerifiedGalleryItemPage, GALLERY_PAGE_SIZE, galleryBoardsOptions } from '@features/gallery/queries';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getProjectWidgetValues } from '@workbench/widgetState';

/** Inject workbench dependencies so reveal loads on demand without adding gallery transfer code to editor boot. */
export interface GalleryRevealContext {
  commands: WorkbenchCommands;
  queries: WorkbenchQueries;
  queryClient: QueryClient;
}

/**
 * What the gesture was, at the moment it was made. The caller captures the
 * account, project, and sequence before lazy loading; the reveal must not take
 * any of them from whatever state exists when the chunk lands.
 */
export interface GalleryRevealTicket {
  /** Account identity lifetime at the original gesture, before any lazy import or media fetch. */
  accountScope: AccountScope;
  /** Canceled when a later Gallery navigation supersedes this locator. */
  locatorSignal?: AbortSignal;
  /** The project the press belongs to; its writes may not land in another. */
  projectId: string;
  /** This navigation's place in the global ordering; see `claimGalleryNavigationSequence`. */
  sequence: number;
}

/**
 * Reveal a freshly resolved item in its board/view after its exact filtered-list position and page agree. Do not
 * change Gallery state until the page verifies the locator result. Caller-minted gesture tickets fence project
 * changes and later selections even across lazy loading.
 */
export const revealGalleryItem = (
  { commands, queries, queryClient }: GalleryRevealContext,
  ref: GalleryItemRef,
  { accountScope, locatorSignal, projectId, sequence }: GalleryRevealTicket
): Promise<void> => {
  // Fence the network result to the gesture's project before clearing filters or selecting.
  const isCurrent = () =>
    !accountScope.signal.aborted && isGalleryNavigationCurrent(sequence) && queries.isActiveProject(projectId);

  if (!isCurrent()) {
    return Promise.resolve();
  }

  return galleryItems.resolve(ref).then(async (image) => {
    assertAccountScopeCurrent(accountScope);

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
    // The board list determines whether the destination can present an archived board. The locator itself uses
    // the same filters as the destination's 60-item page and avoids downloading every item name.
    const boardsPromise = queryClient
      .fetchQuery(
        galleryBoardsOptions({
          includeArchived: settings.showArchivedBoards,
          includeDateBoards: settings.showDateBoards,
          orderBy: settings.boardOrderBy,
          orderDir: settings.boardOrderDir,
        })
      )
      // Unknown beats blocked: without the boards list the reveal proceeds as if the board were listable.
      .catch(() => null);
    const [verified, boards] = await Promise.all([
      fetchVerifiedGalleryItemPage(queryClient, listingFilter, ref, accountScope, locatorSignal),
      boardsPromise,
    ]);

    assertAccountScopeCurrent(accountScope);

    if (!isCurrent() || !verified) {
      return;
    }

    const values = getGalleryValues();
    const settingsNow = getGallerySettings(values);

    // The listing's ordering may have changed while the name list was
    // in flight (sort direction); the computed index describes the old
    // ordering, so the page landing is dropped.
    if (settingsNow.imageOrderDir !== settings.imageOrderDir) {
      return;
    }

    // Do not use a hidden board's offset when board resolution will present Uncategorized instead.
    const isBoardListable =
      image.boardId === 'none' ||
      boards === null ||
      boards.length === 0 ||
      boards.some((board) => board.id === image.boardId);
    const boardIndex = isBoardListable ? verified.index : null;

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
      // The verified page is cached, and its own total lets the listing open it past a stale retained total.
      commands.gallery.setPage(page);
    }

    commands.gallery.selectItem(image, projectId, page ?? undefined);
    if (boardIndex === null) {
      requestGalleryItemReveal(toGalleryItemKey(ref), accountScope.signal);
    } else {
      requestGalleryItemReveal(toGalleryItemKey(ref), accountScope.signal, boardIndex);
    }
  });
};
