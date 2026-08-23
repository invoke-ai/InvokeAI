import type { GalleryView } from '@features/gallery';
import type { GalleryItemsFilter } from '@features/gallery/queries';
import type { QueryClient } from '@tanstack/react-query';

import { galleryImages, legacyGeneratedImageToGalleryItem, toGalleryItemKey } from '@features/gallery';
import { getGallerySettings, registerImageCluster, requestGalleryItemReveal } from '@features/gallery/contracts';
import {
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  galleryBoardsOptions,
  galleryItemNamesOptions,
  galleryItemsInfiniteOptions,
} from '@features/gallery/queries';
import { useQueryClient } from '@tanstack/react-query';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';

export interface MapSelectionActions {
  /** Reveal one image: land the gallery on its board, page, and grid cell. */
  selectImage: (imageName: string) => void;
  /** Show the whole cluster in the gallery, the clicked image selected. */
  selectCluster: (primaryImageName: string, imageNames: string[], label: string) => void;
}

/**
 * Module-scoped, deliberately: the thing being guarded — the gallery selection —
 * is global, so the counter has to be too. A `useRef` is per-mount, which leaves
 * a hole whenever the widget unmounts with a hydrate in flight (switching the
 * right-panel tab away and back): the abandoned closure compares against its own
 * dead ref, still passes, and overwrites the newer mount's selection.
 */
let selectionSequence = 0;

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
 * Turns map clicks into gallery navigation. The map only knows names; the
 * selection contract wants a full gallery item, so names are hydrated through
 * the bulk by-names resolver — always fresh, since a cached DTO's star/board
 * state can drift. ONE monotonic sequence spans both selection kinds, so
 * rapid clicks always resolve to the latest click regardless of which mode
 * each went through; a slow fetch can never overwrite a newer selection.
 * Preview follows the gallery selection on its own.
 *
 * A single-image click is a full reveal: the gallery lands on the image's
 * board and view, any search or similarity filter is cleared (the image may
 * not match it), and the image's position in the board's ordering is looked
 * up so the grid can reach it — the page is selected in paginated mode, and
 * in infinite mode the pages down to it are loaded. The grid scrolls to the
 * newly selected item on its own once it is in the loaded window.
 *
 * A cluster click instead behaves like a search: the cluster's members (in
 * proximity order from the clicked point) become the gallery's list via a
 * `cluster` semantic reference, with the clicked image — the list's first
 * entry — as the selection.
 */
export const useMapSelection = (): MapSelectionActions => {
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();

  const selectImage = useCallback(
    (imageName: string) => {
      const sequence = ++selectionSequence;

      galleryImages
        .resolveMany([imageName])
        .then(async (images) => {
          const image = images.at(0);

          if (!image || sequence !== selectionSequence) {
            return;
          }

          const getGalleryValues = () => getProjectWidgetValues(queries.getSnapshot().activeProject, 'gallery');
          const settings = getGallerySettings(getGalleryValues());
          const targetView: GalleryView = image.imageCategory === 'general' ? 'images' : 'assets';
          // The board listing the gallery will show once the reveal below has
          // cleared any search: identical filter shape, so the name list (and
          // the prefetched pages) land in the cache the gallery reads.
          const listingFilter = {
            boardId: image.boardId,
            galleryView: targetView,
            orderDir: settings.imageOrderDir,
            searchTerm: '',
            starredFirst: settings.starredFirst,
          };
          // The image's position within its board's ordering, which is what
          // lets the gallery land on the right page rather than page 0. A
          // failure here only costs the scroll, not the selection. The boards
          // list rides along because the gallery falls back to Uncategorized
          // when the target board is not listable (archived with "show
          // archived" off) — landing on the hidden board's page number there
          // would jump to an unrelated page of the wrong board.
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
            const index = names.items.findIndex((ref) => ref.kind === 'image' && ref.name === imageName);
            const isBoardListable =
              image.boardId === 'none' ||
              boards === null ||
              boards.length === 0 ||
              boards.some((board) => board.id === image.boardId);

            boardIndex = index >= 0 && isBoardListable ? index : null;
          } catch {
            boardIndex = null;
          }

          if (sequence !== selectionSequence) {
            return;
          }

          const values = getGalleryValues();
          const settingsNow = getGallerySettings(values);

          // The listing's ordering may have changed while the name list was
          // in flight (sort direction, starred-first); the computed index
          // describes the old ordering, so the page landing is dropped.
          if (
            settingsNow.imageOrderDir !== settings.imageOrderDir ||
            settingsNow.starredFirst !== settings.starredFirst
          ) {
            boardIndex = null;
          }

          const currentView: GalleryView = values.galleryView === 'assets' ? 'assets' : 'images';
          const hasSearch = typeof values.searchTerm === 'string' && values.searchTerm !== '';

          // Filters would hide the board listing the index was computed
          // against (the image may not match them), so the reveal clears them.
          if (hasSearch || (values.semanticImageQuery !== null && values.semanticImageQuery !== undefined)) {
            commands.widgets.patchValues('gallery', { searchTerm: '', semanticImageQuery: null });
          }

          if (currentView !== targetView) {
            commands.gallery.setView(targetView);
          }

          // Select the image's board before the image. The map spans every
          // accessible board, but `selectGalleryItem` stamps the navigation
          // query from whatever list the gallery is CURRENTLY showing — a
          // cross-board click without this left Preview's next/prev with no
          // cursor. Mirrors the command palette's reveal-in-gallery.
          commands.gallery.selectBoard(image.boardId);

          const page = boardIndex !== null ? Math.floor(boardIndex / GALLERY_PAGE_SIZE) : null;

          if (page !== null && settingsNow.paginationMode === 'paginated') {
            commands.gallery.setPage(page);
          }

          if (
            boardIndex !== null &&
            page !== null &&
            settingsNow.paginationMode === 'infinite' &&
            boardIndex < GALLERY_MAX_ROWS
          ) {
            // Load every page down to the image so the grid can scroll to it;
            // past the window cap the grid cannot reach it either way. Fire
            // and forget: the selection must not wait on page hydration, and
            // the grid's pending reveal settles whenever the item appears.
            void ensureGalleryPagesLoaded(queryClient, listingFilter, page + 1).catch(() => {});
          }

          commands.gallery.selectItem(legacyGeneratedImageToGalleryItem(image), undefined, page ?? undefined);
          requestGalleryItemReveal(toGalleryItemKey({ kind: 'image', name: imageName }));
        })
        .catch(() => {
          // A click on a just-deleted image, or a blip mid-backend-restart,
          // simply leaves the selection unchanged.
        });
    },
    [commands, queries, queryClient]
  );

  const selectCluster = useCallback(
    (primaryImageName: string, imageNames: string[], label: string) => {
      const sequence = ++selectionSequence;

      galleryImages
        .resolveMany([primaryImageName])
        .then((images) => {
          const image = images.at(0);

          if (!image || sequence !== selectionSequence) {
            return;
          }

          // Same reason as `selectImage` above: the selection stamps the
          // navigation query from the board the gallery is showing, so land
          // on the primary image's board first to keep that query coherent.
          commands.gallery.selectBoard(image.boardId);
          // The member list lives in an in-memory registry (it can run to
          // thousands of names); the persisted value keeps only the key. The
          // page reset and search-term clear mirror setSemanticImageQuery in
          // the gallery's own actions.
          const clusterId = registerImageCluster(imageNames, label);

          commands.widgets.patchValues('gallery', {
            galleryPage: 0,
            searchTerm: '',
            semanticImageQuery: { clusterId, kind: 'cluster', label },
          });
          // The clicked image is the proximity ordering's first entry, so it
          // is selected at the top of the cluster view; Preview follows. The
          // reveal brings the grid back to it even when this exact selection
          // is already current (re-clicking the cluster after scrolling away).
          commands.gallery.selectItem(legacyGeneratedImageToGalleryItem(image));
          requestGalleryItemReveal(toGalleryItemKey({ kind: 'image', name: primaryImageName }));
        })
        .catch(() => {
          // Selection is simply left unchanged on hydrate failure.
        });
    },
    [commands]
  );

  return useMemo(() => ({ selectCluster, selectImage }), [selectCluster, selectImage]);
};
