import type {
  CanvasOperationController,
  CanvasOperationSession,
} from '@workbench/canvas-engine/canvasOperationController';
import type {
  CommitRasterFilterResult,
  ExportLayerPixelsResult,
  LayerExportGuard,
} from '@workbench/canvas-engine/engine';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import type { LayerFilterResult } from './layerFilterRunner';

export interface LayerFilterSettings {
  type: string;
  settings: Record<string, unknown>;
}

export interface FilterOperationPreview {
  guard: LayerExportGuard;
  imageName: string;
  rect: Rect;
  width: number;
  height: number;
  origin: { x: number; y: number };
}

export interface FilterOperationSessionState {
  draft: LayerFilterSettings;
  error: string | null;
  initialFilter: LayerFilterSettings | null;
  layerId: string;
  layerType: 'raster' | 'control';
  preview: FilterOperationPreview | null;
  status: 'ready' | 'processing' | 'committing' | 'error';
}

export type FilterCommitTarget = 'apply' | 'raster' | 'control';

export interface FilterOperationSessionDeps {
  controller: CanvasOperationController;
  exportPixels(): Promise<ExportLayerPixelsResult>;
  runFilter(options: {
    filterType: string;
    input: { surface: RasterSurface; rect: Rect };
    settings: Record<string, unknown>;
    signal: AbortSignal;
  }): Promise<LayerFilterResult>;
  publishPreview(imageName: string, rect: Rect, guard: LayerExportGuard): Promise<'shown' | 'missing' | 'stale'>;
  clearPreview(): void;
  isGuardCurrent(guard: LayerExportGuard): boolean;
  makeDurable(imageName: string): Promise<void>;
  commit(options: {
    draft: LayerFilterSettings;
    guard: LayerExportGuard;
    image: { imageName: string; width: number; height: number };
    origin: { x: number; y: number };
    rect: Rect;
    signal: AbortSignal;
    target: FilterCommitTarget;
  }): Promise<CommitRasterFilterResult>;
}

export interface FilterOperationSession {
  getSnapshot(): FilterOperationSessionState;
  subscribe(listener: () => void): () => void;
  updateDraft(draft: LayerFilterSettings): void;
  process(): Promise<void>;
  reset(settings: Record<string, unknown>): void;
  commit(target: FilterCommitTarget): Promise<void>;
  cancel(): void;
  dispose(): void;
}

export interface CreateFilterOperationSessionOptions {
  deps: FilterOperationSessionDeps;
  guard: LayerExportGuard;
  initialFilter: LayerFilterSettings | null;
  initialDraft?: LayerFilterSettings;
  layerType: 'raster' | 'control';
}

const cloneFilter = (filter: LayerFilterSettings | null): LayerFilterSettings | null =>
  filter ? structuredClone(filter) : null;

const sameGuard = (left: LayerExportGuard, right: LayerExportGuard): boolean =>
  left.projectId === right.projectId &&
  left.layerId === right.layerId &&
  left.layer === right.layer &&
  left.cacheVersion === right.cacheVersion &&
  left.documentGeneration === right.documentGeneration;

const message = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const createFilterOperationSession = (
  options: CreateFilterOperationSessionOptions
): FilterOperationSession | null => {
  const { deps, guard, layerType } = options;
  const initialFilter = cloneFilter(options.initialFilter);
  const fallback = options.initialDraft ?? { settings: {}, type: 'canny_edge_detection' };
  let state: FilterOperationSessionState = {
    draft: cloneFilter(initialFilter) ?? fallback,
    error: null,
    initialFilter,
    layerId: guard.layerId,
    layerType,
    preview: null,
    status: 'ready',
  };
  let disposed = false;
  let commitController: AbortController | null = null;
  let commitToken = 0;
  const listeners = new Set<() => void>();

  const publish = (next: FilterOperationSessionState): void => {
    if (disposed) {
      return;
    }
    state = next;
    for (const listener of listeners) {
      try {
        listener();
      } catch {
        // Session transitions must not be interrupted by a faulty view subscriber.
      }
    }
  };
  const cleanupPreview = (): void => {
    deps.clearPreview();
    if (state.preview) {
      publish({ ...state, preview: null });
    }
  };
  const operation: CanvasOperationSession | null = deps.controller.start({
    cleanupPreview,
    guard,
    identity: { kind: 'filter', layerId: guard.layerId, projectId: guard.projectId },
  });
  if (!operation) {
    return null;
  }

  const process = async (): Promise<void> => {
    if (disposed) {
      return;
    }
    const requestDraft = structuredClone(state.draft);
    publish({ ...state, error: null, preview: null, status: 'processing' });
    const result = await operation.run(
      async (signal) => {
        const exported = await deps.exportPixels();
        if (exported.status !== 'ok') {
          throw new Error(`The filter source is ${exported.status}.`);
        }
        if (!sameGuard(exported.guard, guard) || !deps.isGuardCurrent(guard)) {
          throw new DOMException('The filter source became stale.', 'AbortError');
        }
        const filtered = await deps.runFilter({
          filterType: requestDraft.type,
          input: { rect: exported.rect, surface: exported.surface },
          settings: requestDraft.settings,
          signal,
        });
        if (signal.aborted || !deps.isGuardCurrent(guard)) {
          throw new DOMException('The filter request was superseded.', 'AbortError');
        }
        const rect = { height: filtered.height, width: filtered.width, ...filtered.origin };
        const shown = await deps.publishPreview(filtered.imageName, rect, guard);
        if (signal.aborted || !deps.isGuardCurrent(guard)) {
          throw new DOMException('The filter request was superseded.', 'AbortError');
        }
        if (shown !== 'shown') {
          throw new DOMException('The filter source became stale.', 'AbortError');
        }
        return {
          guard,
          height: filtered.height,
          imageName: filtered.imageName,
          origin: filtered.origin,
          rect,
          width: filtered.width,
        } satisfies FilterOperationPreview;
      },
      (preview) => {
        publish({ ...state, error: null, preview, status: 'ready' });
        return undefined;
      }
    );
    if (result === 'error') {
      const operationState = deps.controller.getSnapshot();
      publish({
        ...state,
        error: operationState.status === 'active' ? operationState.error : 'The filter failed.',
        preview: null,
        status: 'error',
      });
    }
  };

  const cancelCommit = (): void => {
    commitToken += 1;
    commitController?.abort();
    commitController = null;
  };

  const commit = async (target: FilterCommitTarget): Promise<void> => {
    const preview = state.preview;
    if (disposed || !preview || state.status === 'processing' || state.status === 'committing') {
      return;
    }
    cancelCommit();
    const token = commitToken;
    const controller = new AbortController();
    commitController = controller;
    publish({ ...state, error: null, status: 'committing' });
    try {
      await deps.makeDurable(preview.imageName);
      if (disposed || token !== commitToken || controller.signal.aborted || !deps.isGuardCurrent(guard)) {
        return;
      }
      const result = await deps.commit({
        draft: structuredClone(state.draft),
        guard,
        image: { height: preview.height, imageName: preview.imageName, width: preview.width },
        origin: preview.origin,
        rect: preview.rect,
        signal: controller.signal,
        target,
      });
      if (disposed || token !== commitToken || controller.signal.aborted) {
        return;
      }
      if (result.status === 'committed') {
        operation.cancel();
        publish({ ...state, error: null, preview: null, status: 'ready' });
        return;
      }
      publish({
        ...state,
        error: result.status === 'failed' ? result.message : `Filter commit is ${result.status}.`,
        status: 'error',
      });
    } catch (error) {
      if (!disposed && token === commitToken && !controller.signal.aborted) {
        publish({ ...state, error: message(error), status: 'error' });
      }
    } finally {
      if (token === commitToken) {
        commitController = null;
      }
    }
  };

  const cancel = (): void => {
    if (disposed) {
      return;
    }
    cancelCommit();
    operation.cancel();
    publish({ ...state, error: null, preview: null, status: 'ready' });
  };

  return {
    cancel,
    commit,
    dispose: () => {
      if (disposed) {
        return;
      }
      cancelCommit();
      operation.cancel();
      disposed = true;
      listeners.clear();
    },
    getSnapshot: () => state,
    process,
    reset: (settings) => {
      if (disposed) {
        return;
      }
      cancelCommit();
      operation.reset();
      publish({
        ...state,
        draft: { settings: structuredClone(settings), type: state.draft.type },
        error: null,
        preview: null,
        status: 'ready',
      });
    },
    subscribe: (listener) => {
      if (!disposed) {
        listeners.add(listener);
      }
      return () => listeners.delete(listener);
    },
    updateDraft: (draft) => {
      if (disposed) {
        return;
      }
      cancelCommit();
      operation.reset();
      publish({ ...state, draft: structuredClone(draft), error: null, preview: null, status: 'ready' });
    },
  };
};
