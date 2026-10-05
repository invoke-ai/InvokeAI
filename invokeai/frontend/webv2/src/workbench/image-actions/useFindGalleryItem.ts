import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryRevealTicket } from '@workbench/image-actions/revealGalleryItem';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';

import { claimGalleryNavigationSequence, isGalleryNavigationCurrent } from '@features/gallery/contracts';
import { useQueryClient } from '@tanstack/react-query';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';

/**
 * Raise the gallery in its existing region to avoid duplicate placement. Prefer rails over center, where Preview
 * competes; absent grids use default placement.
 */
const SIDE_REGIONS: readonly WidgetRegion[] = ['right', 'left', 'bottom', 'center'];

const getGalleryRegions = (project: Project): WidgetRegion[] =>
  SIDE_REGIONS.filter((region) =>
    project.widgetRegions[region].instanceIds.some(
      (instanceId) => project.widgetInstances[instanceId]?.typeId === 'gallery'
    )
  );

/** Where a reveal that leaves the center alone may raise the Gallery. */
const PANEL_REGIONS: readonly WidgetRegion[] = ['right', 'left', 'bottom'];

export interface FindGalleryItemOptions {
  /**
   * Also bring Preview to the front of the center (the default). Off, the center keeps what it shows, so an editor
   * there stays in front, and the Gallery is raised only in a panel.
   */
  revealPreview?: boolean;
}

/**
 * Raise Preview then Gallery immediately so Gallery wins shared regions, then lazy-load reveal. Claim the gesture
 * ticket before import to preserve press order and project fencing.
 */
export const useFindGalleryItem = (): ((ref: GalleryItemRef, options?: FindGalleryItemOptions) => void) => {
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  const openWorkbenchWidget = useOpenWorkbenchWidget();

  return useCallback(
    (ref: GalleryItemRef, { revealPreview = true }: FindGalleryItemOptions = {}) => {
      const activeProject = queries.getSnapshot().activeProject;
      // Minted here, not inside the import below: both the ordering and the
      // project fence describe the PRESS, and reading either after the chunk
      // lands would pin whatever the workspace had become by then.
      const ticket: GalleryRevealTicket = {
        projectId: activeProject.id,
        sequence: claimGalleryNavigationSequence(),
      };
      const galleryRegions = getGalleryRegions(activeProject);

      if (revealPreview) {
        openWorkbenchWidget('preview', { preferredRegions: ['center'], requireCenterView: true });
        openWorkbenchWidget('gallery', galleryRegions.length > 0 ? { preferredRegions: galleryRegions } : undefined);
      } else {
        const panelRegions = galleryRegions.filter((region) => region !== 'center');

        openWorkbenchWidget('gallery', { preferredRegions: panelRegions.length > 0 ? panelRegions : PANEL_REGIONS });
      }

      void import('@workbench/image-actions/revealGalleryItem')
        .then(({ revealGalleryItem }) => revealGalleryItem({ commands, queries, queryClient }, ref, ticket))
        .catch((error: unknown) => {
          // Report chunk or media failures only for the still-current gesture, since widgets already moved. Keep
          // the failed gesture's claim so older intents cannot reclaim selection.
          if (!isGalleryNavigationCurrent(ticket.sequence) || !queries.isActiveProject(ticket.projectId)) {
            return;
          }

          commands.notifications.reportError({
            area: 'find-in-gallery',
            message: error instanceof Error ? error.message : String(error),
            namespace: 'gallery',
          });
        });
    },
    [commands, openWorkbenchWidget, queries, queryClient]
  );
};
