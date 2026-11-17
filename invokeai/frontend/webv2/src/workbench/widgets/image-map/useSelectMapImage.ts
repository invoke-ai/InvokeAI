import type { GalleryItemKey, GalleryItemRef } from '@features/gallery/contracts';

import { galleryItems, toGalleryItemKey } from '@features/gallery';
import {
  claimGalleryNavigationSequence,
  isGalleryNavigationCurrent,
  parseGallerySemanticReference,
  registerImageCluster,
  requestGalleryItemReveal,
} from '@features/gallery/contracts';
import { abortGalleryLocatorRequests, createGalleryLocatorRequest } from '@features/gallery/utility';
import { captureAccountScope } from '@platform/state/accountLifecycle';
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

/** Single clicks reveal in the gallery, cluster clicks list neighbours; the latest click wins. */
export const useMapSelection = (): MapSelectionActions => {
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  const selectItem = useCallback(
    (ref: GalleryItemRef) => {
      const sequence = claimGalleryNavigationSequence();
      abortGalleryLocatorRequests();
      const locatorRequest = createGalleryLocatorRequest();
      const ticket = {
        accountScope: captureAccountScope(),
        locatorSignal: locatorRequest.signal,
        projectId: queries.getSnapshot().activeProject.id,
        sequence,
      };

      void revealGalleryItem({ commands, queries, queryClient }, ref, ticket)
        .catch(() => {
          // A click on a just-deleted point, or a blip mid-backend-restart, simply leaves the selection unchanged.
        })
        .finally(locatorRequest.release);
    },
    [commands, queries, queryClient]
  );

  const selectCluster = useCallback(
    (primaryItem: GalleryItemRef, itemKeys: GalleryItemKey[], label: string) => {
      const accountScope = captureAccountScope();
      const projectId = queries.getSnapshot().activeProject.id;
      const sequence = claimGalleryNavigationSequence();
      abortGalleryLocatorRequests();
      // Registered while the primary item hydrates, so a reveal nobody asked for cannot supersede this click.
      const locatorRequest = createGalleryLocatorRequest();

      galleryItems
        .resolve(primaryItem)
        .then((image) => {
          if (
            accountScope.signal.aborted ||
            !isGalleryNavigationCurrent(sequence) ||
            !queries.isActiveProject(projectId)
          ) {
            return;
          }

          // Same reason as a single-item reveal: the selection stamps the navigation query from the board the gallery
          // is showing, so land on the primary item's board first to keep that query coherent.
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
          requestGalleryItemReveal(toGalleryItemKey(primaryItem), accountScope.signal);
        })
        .catch(() => {
          // Selection is simply left unchanged on hydrate failure.
        })
        .finally(locatorRequest.release);
    },
    [commands, queries]
  );

  return useMemo(() => ({ selectCluster, selectItem }), [selectCluster, selectItem]);
};

/** Ends a cluster selection only while the gallery shows one: Esc here must not wipe an ordinary search. */
export const useClearClusterSelection = (): (() => void) => {
  const { widgets } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();

  return useCallback(() => {
    const galleryValues = getProjectWidgetValues(queries.getSnapshot().activeProject, 'gallery');

    if (parseGallerySemanticReference(galleryValues.semanticImageQuery)?.kind !== 'cluster') {
      return;
    }

    claimGalleryNavigationSequence();
    abortGalleryLocatorRequests();
    widgets.patchValues('gallery', {
      galleryPage: 0,
      searchTerm: '',
      semanticImageQuery: null,
      semanticSearchText: null,
    });
  }, [queries, widgets]);
};
