import type {
  CanvasOperationController,
  CanvasOperationRunResult,
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

/** Delay between the last draft edit and the automatic preview run. */
export const FILTER_AUTO_PROCESS_DEBOUNCE_MS = 400;

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
  autoProcess: boolean;
  draft: LayerFilterSettings;
  error: string | null;
  initialFilter: LayerFilterSettings | null;
  layerId: string;
  layerName: string;
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
  publishPreview(
    imageName: string,
    rect: Rect,
    guard: LayerExportGuard,
    filterType: string
  ): Promise<'shown' | 'missing' | 'stale'>;
  clearPreview(): void;
  isGuardCurrent(guard: LayerExportGuard): boolean;
  isDraftValid(draft: LayerFilterSettings): boolean;
  makeDurable(imageName: string): Promise<void>;
  canCommit(): boolean;
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
  setAutoProcess(value: boolean): void;
  process(): Promise<CanvasOperationRunResult>;
  interruptProcessing(): void;
  reset(settings: Record<string, unknown>): void;
  /** On 'committed' the canvas operation stays active; the owner must dispose the session to end it. */
  commit(target: FilterCommitTarget): Promise<'committed' | 'blocked' | 'stale'>;
  blockCommit(): void;
  cancel(): void;
  dispose(): void;
}

export interface CreateFilterOperationSessionOptions {
  deps: FilterOperationSessionDeps;
  guard: LayerExportGuard;
  initialFilter: LayerFilterSettings | null;
  initialDraft?: LayerFilterSettings;
  layerName?: string;
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
    autoProcess: true,
    draft: cloneFilter(initialFilter) ?? fallback,
    error: null,
    initialFilter,
    layerId: guard.layerId,
    layerName: options.layerName ?? guard.layer.name,
    layerType,
    preview: null,
    status: 'ready',
  };
  let disposed = false;
  let commitController: AbortController | null = null;
  let commitToken = 0;
  const listeners = new Set<() => void>();
  let autoProcessTimer: ReturnType<typeof setTimeout> | null = null;
  const clearAutoProcess = (): void => {
    if (autoProcessTimer !== null) {
      clearTimeout(autoProcessTimer);
      autoProcessTimer = null;
    }
  };
  const scheduleAutoProcess = (): void => {
    clearAutoProcess();
    if (!state.autoProcess || !deps.isDraftValid(state.draft)) {
      return;
    }
    autoProcessTimer = setTimeout(() => {
      autoProcessTimer = null;
      void process();
    }, FILTER_AUTO_PROCESS_DEBOUNCE_MS);
  };

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

  const process = async (): Promise<CanvasOperationRunResult> => {
    if (disposed) {
      return 'stale';
    }
    clearAutoProcess();
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
        const shown = await deps.publishPreview(filtered.imageName, rect, guard, requestDraft.type);
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
    return result;
  };

  const cancelCommit = (): void => {
    commitToken += 1;
    commitController?.abort();
    commitController = null;
  };

  const commit = async (target: FilterCommitTarget): Promise<'committed' | 'blocked' | 'stale'> => {
    const preview = state.preview;
    if (disposed || !preview || state.status === 'processing' || state.status === 'committing') {
      return 'stale';
    }
    if (!deps.canCommit()) {
      return 'blocked';
    }
    clearAutoProcess();
    cancelCommit();
    const token = commitToken;
    const controller = new AbortController();
    commitController = controller;
    publish({ ...state, error: null, status: 'committing' });
    try {
      if (!deps.canCommit()) {
        publish({ ...state, error: null, status: 'ready' });
        return 'blocked';
      }
      await deps.makeDurable(preview.imageName);
      if (!deps.canCommit()) {
        if (!disposed && token === commitToken) {
          publish({ ...state, error: null, status: 'ready' });
        }
        return 'blocked';
      }
      if (disposed || token !== commitToken || controller.signal.aborted || !deps.isGuardCurrent(guard)) {
        return 'stale';
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
      if (result.status === 'committed') {
        // The operation stays active: the owner ends it by disposing the
        // session, keeping teardown out of this call stack.
        publish({ ...state, error: null, preview: null, status: 'ready' });
        return 'committed';
      }
      if (disposed || token !== commitToken || controller.signal.aborted) {
        return 'stale';
      }
      publish({
        ...state,
        error: result.status === 'failed' ? result.message : `Filter commit is ${result.status}.`,
        status: 'error',
      });
      return result.status === 'locked' ? 'blocked' : 'stale';
    } catch (error) {
      if (!disposed && token === commitToken && !controller.signal.aborted) {
        publish({ ...state, error: message(error), status: 'error' });
      }
      return 'stale';
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
    clearAutoProcess();
    cancelCommit();
    operation.cancel();
    publish({ ...state, error: null, preview: null, status: 'ready' });
  };

  return {
    blockCommit: () => {
      if (commitController) {
        cancelCommit();
        publish({ ...state, error: null, status: 'ready' });
      }
    },
    cancel,
    commit,
    dispose: () => {
      if (disposed) {
        return;
      }
      clearAutoProcess();
      cancelCommit();
      operation.cancel();
      disposed = true;
      listeners.clear();
    },
    getSnapshot: () => state,
    interruptProcessing: () => {
      if (disposed) {
        return;
      }
      clearAutoProcess();
      if (state.status !== 'processing') {
        return;
      }
      operation.interruptProcessing();
      publish({ ...state, error: null, preview: null, status: 'ready' });
    },
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
      scheduleAutoProcess();
    },
    setAutoProcess: (value) => {
      if (disposed || state.autoProcess === value) {
        return;
      }
      publish({ ...state, autoProcess: value });
      if (!value) {
        clearAutoProcess();
        return;
      }
      if (!state.preview && state.status !== 'processing' && state.status !== 'committing') {
        scheduleAutoProcess();
      }
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
      scheduleAutoProcess();
    },
  };
};
