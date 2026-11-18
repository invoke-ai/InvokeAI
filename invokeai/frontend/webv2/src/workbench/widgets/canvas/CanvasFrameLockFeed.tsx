import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { Project } from '@workbench/projectContracts';

import { useMountEffect } from '@platform/react/useMountEffect';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchQueries, useWorkbenchSubscription } from '@workbench/WorkbenchContext';

/**
 * The Frame tool's aspect lock while the canvas drives Generate: Generate's own lock (`aspectRatioIsLocked`) and its
 * ratio. Null when the canvas does not drive Generate; the Frame tool's lock is then its own.
 */
export const readCanvasFrameLock = (project: Project): { aspectLocked: boolean; aspectRatio: number } | null => {
  if (project.invocation.sourceId !== 'canvas') {
    return null;
  }
  const values = getProjectWidgetValues(project, 'generate');
  const width = typeof values.width === 'number' ? values.width : 0;
  const height = typeof values.height === 'number' ? values.height : 0;
  const stored = typeof values.aspectRatioValue === 'number' ? values.aspectRatioValue : Number.NaN;
  return {
    aspectLocked: values.aspectRatioIsLocked === true,
    aspectRatio: Number.isFinite(stored) && stored > 0 ? stored : height > 0 ? width / height : 1,
  };
};

/** Feeds Generate's aspect lock into the Frame tool while the canvas drives Generate, so the two are one lock. */
export const CanvasFrameLockFeed = ({
  engine,
  projectId,
}: {
  engine: Pick<CanvasEngineHandle, 'interaction'>;
  projectId: string;
}) => {
  const queries = useWorkbenchQueries();
  const subscribe = useWorkbenchSubscription();
  useMountEffect(() => {
    let fed: string | null = null;
    const feed = (): void => {
      const project = queries.getProject(projectId);
      const lock = project ? readCanvasFrameLock(project) : null;
      const key = lock ? `${lock.aspectLocked}:${lock.aspectRatio}` : null;
      if (lock && key !== fed) {
        engine.interaction.set('bboxOptions', lock);
      }
      fed = key;
    };
    feed();
    return subscribe(feed);
  });
  return null;
};
