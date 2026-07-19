import type { Viewport } from '@xyflow/react';

const viewports = new Map<string, Viewport>();

export const getWorkflowViewportKey = (projectId: string, instanceId: string): string => `${projectId}:${instanceId}`;

export const getWorkflowViewport = (key: string): Viewport | null => {
  const viewport = viewports.get(key);

  return viewport ? { ...viewport } : null;
};

export const setWorkflowViewport = (key: string, viewport: Viewport): void => {
  viewports.set(key, { x: viewport.x, y: viewport.y, zoom: viewport.zoom });
};
