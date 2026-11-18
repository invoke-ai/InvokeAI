import type { StagedRegion } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { Project } from '@workbench/projectContracts';

import { useMountEffect } from '@platform/react/useMountEffect';
import { getCanvasStagingSlotRect, getCanvasStagingSlots } from '@workbench/canvasStagingView';
import { useWorkbenchQueries, useWorkbenchSubscription } from '@workbench/WorkbenchContext';

/** Where each pending result of `project` lands (staged candidates and results still generating), in slot order. */
export const readCanvasStagedRegions = (project: Pick<Project, 'canvas' | 'queue'>): StagedRegion[] => {
  const { selectedImageIndex } = project.canvas.stagingArea;
  return getCanvasStagingSlots(project.canvas, project.queue.items).map((slot, index) => ({
    rect: getCanvasStagingSlotRect(slot),
    selected: index === selectedImageIndex,
  }));
};

/**
 * Feeds the engine's staged-region overlay, so the canvas shows where pending results will land while the user keeps
 * editing and moves the bbox elsewhere. Rereads only when staging, the revision or the queue changed.
 */
export const CanvasStagedRegionsFeed = ({
  engine,
  projectId,
}: {
  engine: Pick<CanvasEngineHandle, 'previews'>;
  projectId: string;
}) => {
  const queries = useWorkbenchQueries();
  const subscribe = useWorkbenchSubscription();
  useMountEffect(() => {
    let read: { staging: unknown; revision: number; items: unknown } | null = null;
    const feed = (): void => {
      const project = queries.getProject(projectId);
      if (!project) {
        return;
      }
      const { canvas, queue } = project;
      if (
        read &&
        read.staging === canvas.stagingArea &&
        read.revision === canvas.documentRevision &&
        read.items === queue.items
      ) {
        return;
      }
      read = { items: queue.items, revision: canvas.documentRevision, staging: canvas.stagingArea };
      engine.previews.setStagedRegions(readCanvasStagedRegions(project));
    };
    feed();
    // No clearing on unmount: the regions are the project's, and another canvas on this engine may still show them.
    return subscribe(feed);
  });
  return null;
};
