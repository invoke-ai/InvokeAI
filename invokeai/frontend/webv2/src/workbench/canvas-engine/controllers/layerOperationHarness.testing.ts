import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';

import { createHistory } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';

import { createCanvasMutationContext } from './mutationContext';

export interface LayerOperationHarnessOptions {
  /** History byte budget; an edit whose entry can never fit is refused. */
  readonly historyBytes?: number;
  readonly gestureActive?: boolean;
}

/**
 * Layer operations over the real reducer, mutation context, history and layer caches. The mirror is the reducer
 * document; `rasterBytes`, `refuse` and `interleave` let a test starve memory, make the reducer reject a mutation, or
 * break an accepted mutation's postconditions.
 */
export const createLayerOperationHarness = (
  document: CanvasDocumentContractV3,
  options: LayerOperationHarnessOptions = {}
) => {
  const backend = createTestStubRasterBackend();
  const layers = createLayerCacheStore(backend);
  const history = createHistory({ byteBudget: options.historyBytes });
  let project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
    document,
    type: 'replaceCanvasDocument',
  });
  const listeners = new Set<() => void>();
  const dispatched: CanvasProjectMutation[] = [];
  const installed: PreparedLayerCacheReplacement[] = [];
  const state = {
    gestureActive: options.gestureActive ?? false,
    rasterBytes: Number.POSITIVE_INFINITY,
    refuse: (_mutation: CanvasProjectMutation): boolean => false,
    /** A mutation applied right after an accepted one, as an interleaved edit would, to fail its postconditions. */
    interleave: (_mutation: CanvasProjectMutation): CanvasProjectMutation | null => null,
  };
  let reserved = 0;
  let nextId = 0;
  const ctx = createCanvasMutationContext({
    commitEdit: () => undefined,
    createLayerId: () => `new-${(nextId += 1)}`,
    dispatch: (action) => {
      dispatched.push(action);
      if (!state.refuse(action)) {
        project = applyCanvasProjectMutation(project, action);
        const interleaved = state.interleave(action);
        if (interleaved) {
          project = applyCanvasProjectMutation(project, interleaved);
        }
        for (const listener of listeners) {
          listener();
        }
      }
      return true;
    },
    editOwner: Symbol('harness'),
    editingLocked: { get: () => false, subscribe: () => () => undefined },
    getDocument: () => project.canvas.document,
    getReducerDocument: () => project.canvas.document,
    history,
    installPrepared: (prepared) => {
      installed.push(prepared);
      layers.installReplacement(prepared);
    },
    isGestureActive: () => state.gestureActive,
    isGuardCurrent: () => true,
    preparePixels: (layerId, rect, pixels) => layers.prepareReplacement(layerId, rect, pixels),
    projectId: project.id,
    refreshMirror: () => undefined,
    reserveRaster: (bytes) => {
      if (reserved + bytes > state.rasterBytes) {
        return { availableBytes: state.rasterBytes - reserved, requestedBytes: bytes, status: 'over-budget' };
      }
      reserved += bytes;
      let released = false;
      return {
        lease: {
          release: () => {
            if (!released) {
              released = true;
              reserved -= bytes;
            }
          },
        },
        status: 'ok',
      };
    },
    subscribeReducer: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  });
  return {
    backend,
    ctx,
    dispatched,
    document: () => project.canvas.document,
    history,
    installed,
    layers,
    reservedBytes: () => reserved,
    state,
  };
};
