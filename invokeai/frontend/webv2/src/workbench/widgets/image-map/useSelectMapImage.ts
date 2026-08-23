import type { GalleryView } from '@features/gallery';

import { galleryImages, legacyGeneratedImageToGalleryItem } from '@features/gallery';
import { getGallerySettings, registerImageCluster } from '@features/gallery/contracts';
import {
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
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
          // failure here only costs the scroll, not the selection.
          let boardIndex: number | null = null;

          try {
            const names = await queryClient.fetchQuery(galleryItemNamesOptions(listingFilter));
            const index = names.items.findIndex((ref) => ref.kind === 'image' && ref.name === imageName);

            boardIndex = index >= 0 ? index : null;
          } catch {
            boardIndex = null;
          }

          if (sequence !== selectionSequence) {
            return;
          }

          const values = getGalleryValues();
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

          if (page !== null && settings.paginationMode === 'paginated') {
            commands.gallery.setPage(page);
          }

          if (
            boardIndex !== null &&
            page !== null &&
            settings.paginationMode === 'infinite' &&
            boardIndex < GALLERY_MAX_ROWS
          ) {
            // Load every page down to the image so the grid can scroll to it;
            // past the window cap the grid cannot reach it either way. Fire
            // and forget: the selection must not wait on page hydration, and
            // the grid scrolls whenever the item appears.
            void queryClient.prefetchInfiniteQuery({
              ...galleryItemsInfiniteOptions(listingFilter, { kind: 'infinite' }),
              pages: page + 1,
            });
          }

          commands.gallery.selectItem(legacyGeneratedImageToGalleryItem(image), undefined, page ?? undefined);
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
          // is selected at the top of the cluster view; Preview follows.
          commands.gallery.selectItem(legacyGeneratedImageToGalleryItem(image));
        })
        .catch(() => {
          // Selection is simply left unchanged on hydrate failure.
        });
    },
    [commands]
  );

  return useMemo(() => ({ selectCluster, selectImage }), [selectCluster, selectImage]);
};
