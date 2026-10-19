import type { GalleryView } from '@features/gallery';
import type { GalleryItemRef, GalleryRevealAnnouncement } from '@features/gallery/contracts';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';

import { galleryItems, toGalleryItemKey } from '@features/gallery';
import {
  getGallerySettings,
  isGalleryNavigationCurrent,
  requestGalleryItemReveal,
  toGalleryMediaKind,
} from '@features/gallery/contracts';
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
  /** Announced in the Gallery's live region once the item is revealed; reveals the user asked for set none. */
  announcement?: GalleryRevealAnnouncement;
  /** Canceled when a later Gallery navigation supersedes this locator. */
  locatorSignal?: AbortSignal;
  /**
   * A further condition checked wherever the ticket is, last immediately before the first Gallery change; false
   * stands the reveal down. For reveals nobody on screen asked for.
   */
  mayApply?: () => boolean;
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
  { accountScope, announcement, locatorSignal, mayApply, projectId, sequence }: GalleryRevealTicket
): Promise<void> => {
  // Fence the network result to the gesture's project before clearing filters or selecting.
  const isCurrent = () =>
    !accountScope.signal.aborted &&
    isGalleryNavigationCurrent(sequence) &&
    queries.isActiveProject(projectId) &&
    (mayApply?.() ?? true);

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
    // A media filter that hides the item's kind is reset to both, like the other filters; one that shows it stays.
    const currentMediaKind = toGalleryMediaKind(settings.mediaFilter);
    const resetsMediaFilter = currentMediaKind !== undefined && currentMediaKind !== image.kind;
    const listingFilter = {
      boardId: image.boardId,
      galleryView: targetView,
      ...(currentMediaKind && !resetsMediaFilter ? { mediaKind: currentMediaKind } : {}),
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

    // The listing's ordering or media filter may have changed while the locator was in flight; the computed index
    // describes the old listing, so the page landing is dropped.
    if (settingsNow.imageOrderDir !== settings.imageOrderDir || settingsNow.mediaFilter !== settings.mediaFilter) {
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
      resetsMediaFilter ||
      (values.starredOnly === true) !== wantsStarredOnly ||
      (values.semanticImageQuery !== null && values.semanticImageQuery !== undefined)
    ) {
      commands.widgets.patchValues('gallery', {
        ...(resetsMediaFilter ? { mediaFilter: 'all' } : {}),
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
    if (announcement !== undefined) {
      requestGalleryItemReveal(toGalleryItemKey(ref), accountScope.signal, boardIndex ?? undefined, announcement);
    } else if (boardIndex === null) {
      requestGalleryItemReveal(toGalleryItemKey(ref), accountScope.signal);
    } else {
      requestGalleryItemReveal(toGalleryItemKey(ref), accountScope.signal, boardIndex);
    }
  });
};
