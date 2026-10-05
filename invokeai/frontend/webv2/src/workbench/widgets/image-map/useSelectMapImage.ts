import type { GalleryItemKey, GalleryItemRef } from '@features/gallery/contracts';

import { galleryItems, toGalleryItemKey } from '@features/gallery';
import {
  claimGalleryNavigationSequence,
  isGalleryNavigationCurrent,
  parseGallerySemanticReference,
  registerImageCluster,
  requestGalleryItemReveal,
} from '@features/gallery/contracts';
import { useQueryClient } from '@tanstack/react-query';
import { revealGalleryItem } from '@workbench/image-actions/revealGalleryItem';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';

export interface MapSelectionActions {
  /** Reveal one item: land the gallery on its board, page, and grid cell. */
  selectItem: (item: GalleryItemRef) => void;
  /** Show the whole cluster in the gallery, the clicked item selected. */
  selectCluster: (primaryItem: GalleryItemRef, itemKeys: GalleryItemKey[], label: string) => void;
}

/**
 * Single clicks use shared gallery reveal without raising widgets. Cluster clicks populate a proximity-ordered
 * semantic list. Both share gesture ordering so the latest click wins.
 */
export const useMapSelection = (): MapSelectionActions => {
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  const selectItem = useCallback(
    (ref: GalleryItemRef) => {
      const ticket = { projectId: queries.getSnapshot().activeProject.id, sequence: claimGalleryNavigationSequence() };

      void revealGalleryItem({ commands, queries, queryClient }, ref, ticket).catch(() => {
        // A click on a just-deleted point, or a blip mid-backend-restart, simply
        // leaves the selection unchanged. The map sits beside the grid and has
        // moved nothing, so there is nothing to explain.
      });
    },
    [commands, queries, queryClient]
  );

  const selectCluster = useCallback(
    (primaryItem: GalleryItemRef, itemKeys: GalleryItemKey[], label: string) => {
      const sequence = claimGalleryNavigationSequence();

      galleryItems
        .resolve(primaryItem)
        .then((image) => {
          if (!isGalleryNavigationCurrent(sequence)) {
            return;
          }

          // Same reason as a single-item reveal: the selection stamps the
          // navigation query from the board the gallery is showing, so land
          // on the primary item's board first to keep that query coherent.
          commands.gallery.selectBoard(image.boardId);
          // Keep large member lists in memory and persist only their key; reset page/search like gallery semantic
          // queries.
          const clusterId = registerImageCluster(itemKeys, label);

          commands.widgets.patchValues('gallery', {
            galleryPage: 0,
            searchTerm: '',
            semanticImageQuery: { clusterId, kind: 'cluster', label },
            semanticSearchText: null,
          });
          // Select and reveal the proximity list's first item even if already selected, restoring scroll after a
          // repeated click.
          commands.gallery.selectItem(image);
          requestGalleryItemReveal(toGalleryItemKey(primaryItem));
        })
        .catch(() => {
          // Selection is simply left unchanged on hydrate failure.
        });
    },
    [commands]
  );

  return useMemo(() => ({ selectCluster, selectItem }), [selectCluster, selectItem]);
};

/**
 * Ends a cluster selection: the gallery drops the cluster listing (as its own chip's clear does) and the map regains
 * its colours, since both draw from the same reference. Claiming a navigation ticket retires a cluster click still
 * hydrating, which would otherwise re-apply the selection just cleared.
 *
 * Checks at call time that a cluster is what the gallery shows: Esc reaches this whenever the map has focus, and
 * the same reset applied to an ordinary search would wipe the user's search text.
 */
export const useClearClusterSelection = (): (() => void) => {
  const { widgets } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();

  return useCallback(() => {
    const galleryValues = getProjectWidgetValues(queries.getSnapshot().activeProject, 'gallery');

    if (parseGallerySemanticReference(galleryValues.semanticImageQuery)?.kind !== 'cluster') {
      return;
    }

    claimGalleryNavigationSequence();
    widgets.patchValues('gallery', {
      galleryPage: 0,
      searchTerm: '',
      semanticImageQuery: null,
      semanticSearchText: null,
    });
  }, [queries, widgets]);
};
