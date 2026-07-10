import type {
  CommitRasterFilterOptions,
  CommitRasterFilterResult,
  ExportLayerPixelsResult,
  LayerExportGuard,
} from '@workbench/canvas-engine/engine';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import type { LayerFilterResult } from './layerFilterRunner';

export interface LayerFilterControllerDeps {
  exportPixels(): Promise<ExportLayerPixelsResult>;
  runFilter(options: {
    filterType: string;
    input: { surface: RasterSurface; rect: Rect };
    settings: Record<string, unknown>;
    signal: AbortSignal;
  }): Promise<LayerFilterResult>;
  showPreview(imageName: string, guard: LayerExportGuard): Promise<'shown' | 'missing' | 'stale'>;
  clearPreview(): void;
  makeDurable(imageName: string): Promise<void>;
  commit(
    options: Omit<CommitRasterFilterOptions, 'signal'> & { signal: AbortSignal }
  ): Promise<CommitRasterFilterResult>;
}

export interface LayerFilterControllerPreview {
  guard: LayerExportGuard;
  imageName: string;
  rect: Rect;
  width: number;
  height: number;
}

export type LayerFilterControllerError =
  | { key: 'busy' | 'locked' | 'stale' | 'unsupported' }
  | { key: 'commitFailure' | 'durabilityFailure' | 'graphFailure'; message: string };

export interface LayerFilterControllerState {
  isRunning: boolean;
  preview: LayerFilterControllerPreview | null;
  error: LayerFilterControllerError | null;
}

export interface LayerFilterController {
  getSnapshot(): LayerFilterControllerState;
  subscribe(listener: () => void): () => void;
  preview(filterType: string, settings: Record<string, unknown>): Promise<void>;
  commit(mode: 'replace' | 'copy'): Promise<void>;
  cancel(): void;
  dispose(): void;
}

const errorMessage = (cause: unknown): string => (cause instanceof Error ? cause.message : String(cause));

export const createLayerFilterController = (deps: LayerFilterControllerDeps): LayerFilterController => {
  let state: LayerFilterControllerState = { error: null, isRunning: false, preview: null };
  let activeController: AbortController | null = null;
  let requestToken = 0;
  let isDisposed = false;
  const listeners = new Set<() => void>();

  const publish = (nextState: LayerFilterControllerState): void => {
    if (isDisposed) {
      return;
    }
    state = nextState;
    listeners.forEach((listener) => listener());
  };

  const beginRequest = (): { controller: AbortController; token: number } => {
    activeController?.abort();
    const controller = new AbortController();
    requestToken += 1;
    activeController = controller;
    return { controller, token: requestToken };
  };

  const isCurrentRequest = (token: number, controller: AbortController): boolean =>
    !isDisposed && requestToken === token && activeController === controller && !controller.signal.aborted;

  const finishRequest = (token: number, controller: AbortController): void => {
    if (!isCurrentRequest(token, controller)) {
      return;
    }
    activeController = null;
    publish({ ...state, isRunning: false });
  };

  const preview = async (filterType: string, settings: Record<string, unknown>): Promise<void> => {
    const { controller, token } = beginRequest();
    deps.clearPreview();
    publish({ error: null, isRunning: true, preview: null });
    try {
      const exported = await deps.exportPixels();
      if (!isCurrentRequest(token, controller)) {
        return;
      }
      if (exported.status !== 'ok') {
        publish({
          error: {
            key: exported.status === 'unsupported' || exported.status === 'empty' ? 'unsupported' : 'stale',
          },
          isRunning: true,
          preview: null,
        });
        return;
      }

      const result = await deps.runFilter({
        filterType,
        input: { rect: exported.rect, surface: exported.surface },
        settings,
        signal: controller.signal,
      });
      if (!isCurrentRequest(token, controller)) {
        return;
      }
      const shown = await deps.showPreview(result.imageName, exported.guard);
      if (!isCurrentRequest(token, controller)) {
        return;
      }
      if (shown !== 'shown') {
        publish({ error: { key: 'stale' }, isRunning: true, preview: null });
        return;
      }
      publish({
        error: null,
        isRunning: true,
        preview: {
          guard: exported.guard,
          height: result.height,
          imageName: result.imageName,
          rect: exported.rect,
          width: result.width,
        },
      });
    } catch (cause) {
      if (isCurrentRequest(token, controller)) {
        publish({ error: { key: 'graphFailure', message: errorMessage(cause) }, isRunning: true, preview: null });
      }
    } finally {
      finishRequest(token, controller);
    }
  };

  const commit = async (mode: 'replace' | 'copy'): Promise<void> => {
    const candidate = state.preview;
    if (!candidate || isDisposed) {
      return;
    }
    const { controller, token } = beginRequest();
    publish({ ...state, error: null, isRunning: true });
    try {
      try {
        await deps.makeDurable(candidate.imageName);
      } catch (cause) {
        if (isCurrentRequest(token, controller)) {
          publish({
            error: { key: 'durabilityFailure', message: errorMessage(cause) },
            isRunning: true,
            preview: candidate,
          });
        }
        return;
      }
      if (!isCurrentRequest(token, controller)) {
        return;
      }

      let result: CommitRasterFilterResult;
      try {
        result = await deps.commit({
          guard: candidate.guard,
          image: { height: candidate.height, imageName: candidate.imageName, width: candidate.width },
          mode,
          rect: candidate.rect,
          signal: controller.signal,
        });
      } catch (cause) {
        if (isCurrentRequest(token, controller)) {
          publish({
            error: { key: 'commitFailure', message: errorMessage(cause) },
            isRunning: true,
            preview: candidate,
          });
        }
        return;
      }
      if (!isCurrentRequest(token, controller)) {
        return;
      }

      if (result.status === 'committed') {
        deps.clearPreview();
        publish({ error: null, isRunning: true, preview: null });
        return;
      }
      if (result.status === 'aborted') {
        return;
      }
      const error: LayerFilterControllerError =
        result.status === 'locked'
          ? { key: 'locked' }
          : result.status === 'unsupported'
            ? { key: 'unsupported' }
            : result.status === 'busy'
              ? { key: 'busy' }
              : result.status === 'failed'
                ? { key: 'commitFailure', message: result.message }
                : { key: 'stale' };
      publish({ error, isRunning: true, preview: candidate });
    } finally {
      finishRequest(token, controller);
    }
  };

  const cancel = (): void => {
    requestToken += 1;
    activeController?.abort();
    activeController = null;
    deps.clearPreview();
    publish({ error: null, isRunning: false, preview: null });
  };

  const dispose = (): void => {
    if (isDisposed) {
      return;
    }
    isDisposed = true;
    requestToken += 1;
    activeController?.abort();
    activeController = null;
    deps.clearPreview();
    listeners.clear();
  };

  return {
    cancel,
    commit,
    dispose,
    getSnapshot: () => state,
    preview,
    subscribe: (listener) => {
      if (!isDisposed) {
        listeners.add(listener);
      }
      return () => listeners.delete(listener);
    },
  };
};
