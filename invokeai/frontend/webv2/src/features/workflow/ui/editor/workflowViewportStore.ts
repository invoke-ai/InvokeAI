import type { Viewport } from '@xyflow/react';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';

const viewports = new Map<string, Viewport>();

export const clearWorkflowViewports = (): void => {
  viewports.clear();
};

registerAccountOwnedResource({
  clear: clearWorkflowViewports,
  name: 'workflow-viewports',
});

/** One viewport per project workflow per editor instance, so switching workflows brings each one's view back. */
export const getWorkflowViewportKey = (projectId: string, workflowId: string, instanceId: string): string =>
  `${projectId}\u0000${workflowId}\u0000${instanceId}`;

/** Releases the viewports of a project's workflows that no longer exist. */
export const releaseWorkflowViewportsExcept = (projectId: string, liveWorkflowIds: readonly string[]): void => {
  const live = new Set(liveWorkflowIds);

  for (const key of viewports.keys()) {
    const [keyProjectId, keyWorkflowId] = key.split('\u0000');

    if (keyProjectId === projectId && keyWorkflowId !== undefined && !live.has(keyWorkflowId)) {
      viewports.delete(key);
    }
  }
};

export const getWorkflowViewport = (key: string): Viewport | null => {
  const viewport = viewports.get(key);

  return viewport ? { ...viewport } : null;
};

export const setWorkflowViewport = (key: string, viewport: Viewport): void => {
  viewports.set(key, { x: viewport.x, y: viewport.y, zoom: viewport.zoom });
};
