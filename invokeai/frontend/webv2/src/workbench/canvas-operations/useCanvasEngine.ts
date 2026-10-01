import type { CanvasEngine, ImageResolver } from '@workbench/canvas-engine/api';

import { useFontRuntime } from '@features/fonts/react';
import { galleryImageUrls } from '@features/gallery/utility';
import { getModelsSnapshot } from '@features/models';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { publishLayerPanelSelection, readLayerPanelState } from '@workbench/layerPanelState';
import { resolveDefaultControlModelForBase } from '@workbench/widgets/layers/controlModelOptions';
import { getSelectedModelBase } from '@workbench/widgets/layers/selectedModel';
import {
  useActiveProjectId,
  useWorkbenchLiveCanvasEngines,
  useWorkbenchCommands,
  useWorkbenchInternalStore,
  useWorkbenchPersistenceService,
} from '@workbench/WorkbenchContext';
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
  const fonts = useFontRuntime();
  const store = useWorkbenchInternalStore();
  const persistence = useWorkbenchPersistenceService();
  const liveEngines = useWorkbenchLiveCanvasEngines();
  const { notifications } = useWorkbenchCommands();
  const projectId = useActiveProjectId();
  const resource = useMemo(
    () =>
      createCanvasEngineResource(projectId, {
        ensureProjectOnServer: async () => {
          const project = store.getState().projects.find((candidate) => candidate.id === projectId);
          if (!project) {
            throw new DOMException('The canvas project is no longer open.', 'AbortError');
          }
          await persistence.ensureProjectOnServer(project);
          if (!store.getState().projects.some((candidate) => candidate.id === projectId)) {
            throw new DOMException('The canvas project is no longer open.', 'AbortError');
          }
        },
        liveEngines,
        getDefaultControlModel: (base) => resolveDefaultControlModelForBase(getModelsSnapshot().models, base),
        getMainModelBase: () => {
          const project = store.getState().projects.find((candidate) => candidate.id === projectId);
          return project ? getSelectedModelBase(project) : null;
        },
        getSelectedLayerIds: () => {
          const project = store.getState().projects.find((candidate) => candidate.id === projectId);
          return project ? readLayerPanelState(projectId, project.canvas.document.selectedLayerId).selectedIds : [];
        },
        setSelectedLayerIds: (primaryId, selectedIds) =>
          publishLayerPanelSelection({ primaryId, projectId, selectedIds }),
        imageResolver: createImageResolver(),
        fonts,
        mutationPort: createCanvasProjectMutationPort(store, projectId),
        reportError: notifications.reportError,
      }),
    [fonts, liveEngines, notifications.reportError, persistence, projectId, store]
  );

  return useSyncExternalStore(resource.subscribe, resource.getSnapshot, resource.getSnapshot);
};
