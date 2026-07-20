import type { CanvasEngine, ImageResolver } from '@workbench/canvas-engine/api';

import { galleryImageUrls } from '@features/gallery/utility';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { getSelectedModelBase } from '@workbench/widgets/layers/selectedModel';
import { useActiveProjectId, useWorkbenchCommands, useWorkbenchInternalStore } from '@workbench/WorkbenchContext';
import { useMemo, useSyncExternalStore } from 'react';

import type { EngineDeps } from './engineRegistry';

import { getOrCreateEngine, releaseEngine } from './engineRegistry';

export type CanvasEngineHandle = CanvasEngine;

export interface CanvasEngineResource {
  getSnapshot(): CanvasEngine | null;
  subscribe(listener: () => void): () => void;
}

/** Fetches a persisted image asset to a Blob for the engine rasterizers. */
const createImageResolver = (): ImageResolver => async (imageName, signal) => {
  const response = await fetch(galleryImageUrls.full(imageName), { signal });
  if (!response.ok) {
    throw new Error(`Failed to load canvas image "${imageName}" (${response.status})`);
  }
  return response.blob();
};

/**
 * Turns one registry lease into a React external store. The first subscriber
 * acquires the engine and the last subscriber releases it, so speculative
 * renders never create engines and StrictMode subscriptions remain balanced.
 */
export const createCanvasEngineResource = (projectId: string, deps: EngineDeps): CanvasEngineResource => {
  let engine: CanvasEngine | null = null;
  const listeners = new Set<() => void>();

  return {
    getSnapshot: () => engine,
    subscribe: (listener) => {
      listeners.add(listener);
      if (listeners.size === 1) {
        engine = getOrCreateEngine(projectId, deps);
        listener();
      }
      return () => {
        listeners.delete(listener);
        if (listeners.size === 0 && engine) {
          engine = null;
          releaseEngine(projectId);
        }
      };
    },
  };
};

/** Returns the active project's shared engine through a balanced registry lease. */
export const useCanvasEngine = (): CanvasEngineHandle | null => {
  const store = useWorkbenchInternalStore();
  const { notifications } = useWorkbenchCommands();
  const projectId = useActiveProjectId();
  const resource = useMemo(
    () =>
      createCanvasEngineResource(projectId, {
        getMainModelBase: () => {
          const project = store.getState().projects.find((candidate) => candidate.id === projectId);
          return project ? getSelectedModelBase(project) : null;
        },
        imageResolver: createImageResolver(),
        mutationPort: createCanvasProjectMutationPort(store, projectId),
        reportError: notifications.reportError,
      }),
    [notifications.reportError, projectId, store]
  );

  return useSyncExternalStore(resource.subscribe, resource.getSnapshot, resource.getSnapshot);
};
