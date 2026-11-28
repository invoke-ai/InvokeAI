import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryRevealTicket } from '@workbench/image-actions/revealGalleryItem';

import { claimGalleryNavigationSequence, isGalleryNavigationCurrent } from '@features/gallery/contracts';
import { abortGalleryLocatorRequests, createGalleryLocatorRequest } from '@features/gallery/utility';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useWorkspaceOpen } from '@workbench/workspace/useWorkspaceOpen';
import { useCallback } from 'react';

export interface FindGalleryItemOptions {
  /**
   * Also bring Preview to the front of the center (the default). Off, the center keeps what it shows, so an editor
   * there stays in front, and the Gallery is raised only in a panel.
   */
  revealPreview?: boolean;
}

/**
 * Show Preview, then the Gallery beside it, in one gesture, then lazy-load the reveal. Claim the gesture ticket before
 * import to preserve press order and project fencing.
 */
export const useFindGalleryItem = (): ((ref: GalleryItemRef, options?: FindGalleryItemOptions) => void) => {
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  const open = useWorkspaceOpen();

  return useCallback(
    (ref: GalleryItemRef, { revealPreview = true }: FindGalleryItemOptions = {}) => {
      const activeProject = queries.getSnapshot().activeProject;
      // Minted here, not inside the import below: both the ordering and the project fence describe the PRESS, and
      // reading either after the chunk lands would pin whatever the workspace had become by then.
      const sequence = claimGalleryNavigationSequence();
      abortGalleryLocatorRequests();
      const locatorRequest = createGalleryLocatorRequest();
      const ticket: GalleryRevealTicket = {
        accountScope: captureAccountScope(),
        locatorSignal: locatorRequest.signal,
        projectId: activeProject.id,
        sequence,
      };

      open(
        revealPreview
          ? [{ surface: 'stage', typeId: 'preview' }, { typeId: 'gallery' }]
          : { surface: 'panel', typeId: 'gallery' },
        // The grid takes focus, so the keys move around the found item.
        { focus: 'gallery' }
      );

      void import('@workbench/image-actions/revealGalleryItem')
        .then(({ revealGalleryItem }) => revealGalleryItem({ commands, queries, queryClient }, ref, ticket))
        .catch((error: unknown) => {
          // Report chunk or media failures only for the still-current gesture, since widgets already moved. Keep the
          // failed gesture's claim so older intents cannot reclaim selection.
          if (
            ticket.accountScope.signal.aborted ||
            !isGalleryNavigationCurrent(ticket.sequence) ||
            !queries.isActiveProject(ticket.projectId)
          ) {
            return;
          }

          commands.notifications.reportError({
            area: 'find-in-gallery',
            message: error instanceof Error ? error.message : String(error),
            namespace: 'gallery',
          });
        })
        .finally(locatorRequest.release);
    },
    [commands, open, queries, queryClient]
  );
};
