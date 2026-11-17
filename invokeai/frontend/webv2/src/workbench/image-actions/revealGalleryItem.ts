import type { GalleryView } from '@features/gallery';
import type { GalleryItemRef, GalleryRevealAnnouncement } from '@features/gallery/contracts';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';

import { galleryItems, toGalleryItemKey } from '@features/gallery';
import {
  getGalleryItemView,
  getGallerySettings,
  isGalleryNavigationCurrent,
  requestGalleryItemReveal,
  toGalleryMediaKind,
} from '@features/gallery/contracts';
import { fetchVerifiedGalleryItemPage, GALLERY_PAGE_SIZE, galleryBoardsOptions } from '@features/gallery/queries';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';

/** Injected so reveal loads on demand without adding gallery transfer code to editor boot. */
export interface GalleryRevealContext {
  commands: WorkbenchCommands;
  queries: WorkbenchQueries;
  queryClient: QueryClient;
}

/** Captured before lazy loading: the reveal must not read account, project or sequence from later state. */
export interface GalleryRevealTicket {
  accountScope: AccountScope;
  /** Reveals the user asked for set none. */
  announcement?: GalleryRevealAnnouncement;
  locatorSignal?: AbortSignal;
  /** Checked last immediately before the first Gallery change; for reveals nobody on screen asked for. */
  mayApply?: () => boolean;
  projectId: string;
  sequence: number;
}

/** No Gallery state changes before the item's filtered-list position and page agree. */
export const revealGalleryItem = (
  { commands, queries, queryClient }: GalleryRevealContext,
  ref: GalleryItemRef,
  { accountScope, announcement, locatorSignal, mayApply, projectId, sequence }: GalleryRevealTicket
): Promise<void> => {
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

    const targetView = getGalleryItemView(image);

    // A Canvas-owned (`other`) item is in no view's listing.
    if (targetView === null) {
      return;
    }

    const getGalleryValues = () => {
      const snapshot = queries.getSnapshot();
      return getResolvedProjectWidgetValues(snapshot.activeProject, 'gallery', snapshot.scopedDefaults);
    };
    const settings = getGallerySettings(getGalleryValues());
    // Match the post-reveal filters so prefetched pages enter the gallery's cache.
    const wantsStarredOnly = image.starred === true;
    // A media filter that hides the item's kind resets to both.
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
    const boardsPromise = queryClient
      .fetchQuery(
        galleryBoardsOptions({
          includeArchived: settings.showArchivedBoards,
          includeDateBoards: settings.showDateBoards,
          orderBy: settings.boardOrderBy,
          orderDir: settings.boardOrderDir,
        })
      )
      // Unknown beats blocked: without the boards list the reveal proceeds.
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

    // The computed index describes the old listing if ordering or media filter changed meanwhile.
    if (settingsNow.imageOrderDir !== settings.imageOrderDir || settingsNow.mediaFilter !== settings.mediaFilter) {
      return;
    }

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

    // Board before item, so its navigation query captures the destination listing for Preview next/previous.
    commands.gallery.selectBoard(image.boardId);

    const page = boardIndex !== null ? Math.floor(boardIndex / GALLERY_PAGE_SIZE) : null;

    if (page !== null && settingsNow.paginationMode === 'paginated') {
      // Its own total lets the listing open it past a stale retained total.
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
