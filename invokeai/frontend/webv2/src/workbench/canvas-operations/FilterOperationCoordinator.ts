import type { LayerExportGuard } from '@workbench/canvas-engine/api';
import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type {
  CanvasOperationActionResult,
  CanvasOperationMutationResult,
  FilterCommitOperationResult,
  StartFilterOperationResult,
} from '@workbench/canvas-operations/api';
import type {
  CanvasApplicationOperationStores,
  CanvasOperationController,
  CanvasOperationRunResult,
  CreateFilterSessionOptions,
  FilterOperationSession,
} from '@workbench/canvas-operations/contracts';
import type { FilterCommitTarget, LayerFilterSettings } from '@workbench/canvas-operations/operationTypes';
import type { BackendGraphContract } from '@workbench/graphContracts';

import {
  buildFilterDefaults,
  DEFAULT_CONTROL_FILTER_TYPE,
  getFilterDefinition,
  isFilterConfigValid,
} from '@workbench/canvas-operations/filterGraphs';

import { createFilterOperationSession } from './filterOperationSession';
import { runLayerFilter } from './layerFilterRunner';

export interface FilterOperationCoordinatorDeps {
  readonly stores: CanvasApplicationOperationStores;
  readonly controller: CanvasOperationController;
  isInteractionLocked(): boolean;
  getDocument(): CanvasDocumentContractV2 | null;
  captureGuard(layerId: string): LayerExportGuard | null;
  selectLayer(layerId: string): void;
  clearOtherOperation(): void;
  clearPreview(layerId: string): void;
  setViewTool(): void;
  encodeSurface(surface: RasterSurface): Promise<Blob>;
  runFilterGraph(options: {
    graph: BackendGraphContract;
    outputNodeId?: string;
    signal?: AbortSignal;
  }): Promise<{ height: number; imageName: string; width: number }>;
  uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ imageName: string }>;
  createSession?(options: CreateFilterSessionOptions): FilterOperationSession | null;
  getSessionDeps(
    layerId: string
  ): Omit<
    CreateFilterSessionOptions['deps'],
    'canCommit' | 'clearPreview' | 'controller' | 'isDraftValid' | 'makeDurable' | 'runFilter'
  >;
}

export interface FilterOperationCoordinator {
  start(layerId: string, recommendedFilterType?: string | null): StartFilterOperationResult;
  updateDraft(draft: LayerFilterSettings): CanvasOperationMutationResult;
  setAutoProcess(value: boolean): CanvasOperationMutationResult;
  process(): Promise<CanvasOperationActionResult>;
  reset(settings: Record<string, unknown>): CanvasOperationMutationResult;
  commit(
    target: FilterCommitTarget,
    makeImageDurable: (imageName: string) => Promise<void>
  ): Promise<FilterCommitOperationResult>;
  cancel(): void;
  interruptAndBlock(): void;
  isActive(): boolean;
  dispose(): void;
}

export const createFilterOperationCoordinator = (deps: FilterOperationCoordinatorDeps): FilterOperationCoordinator => {
  let session: FilterOperationSession | null = null;
  let unsubscribeSession: (() => void) | null = null;
  let unsubscribeController: (() => void) | null = null;
  let makeDurable: (imageName: string) => Promise<void> = () =>
    Promise.reject(new Error('The filter result cannot be preserved.'));

  const syncStore = (): void => deps.stores.filterSession.set(session?.getSnapshot() ?? null);
  const clear = (): void => {
    const owned = session;
    session = null;
    unsubscribeSession?.();
    unsubscribeSession = null;
    unsubscribeController?.();
    unsubscribeController = null;
    owned?.dispose();
    deps.stores.filterSession.set(null);
  };

  const start = (layerId: string, recommendedFilterType?: string | null): StartFilterOperationResult => {
    if (deps.isInteractionLocked()) {
      return 'locked';
    }
    if (recommendedFilterType && deps.controller.getSnapshot().status !== 'idle') {
      return 'not-ready';
    }
    const document = deps.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
    if (!document || !layer) {
      return 'missing';
    }
    if (layer.type !== 'raster' && layer.type !== 'control') {
      return 'unsupported';
    }
    if (recommendedFilterType && layer.filter) {
      return 'not-ready';
    }
    if (!layer.isEnabled) {
      return 'disabled';
    }
    if (layer.isLocked) {
      return 'locked';
    }
    const guard = deps.captureGuard(layer.id);
    if (!guard) {
      return 'not-ready';
    }
    const initialType = layer.filter?.type ?? recommendedFilterType ?? DEFAULT_CONTROL_FILTER_TYPE;
    const initialFilter = layer.filter ? structuredClone(layer.filter) : null;
    const definition = getFilterDefinition(initialType);
    const draft = initialFilter ?? {
      settings: definition ? buildFilterDefaults(definition) : {},
      type: definition?.type ?? DEFAULT_CONTROL_FILTER_TYPE,
    };

    deps.clearOtherOperation();
    clear();
    if (document.selectedLayerId !== layer.id) {
      deps.selectLayer(layer.id);
    }
    const createSession = deps.createSession ?? createFilterOperationSession;
    session = createSession({
      deps: {
        ...deps.getSessionDeps(layer.id),
        canCommit: () => !deps.isInteractionLocked(),
        clearPreview: () => deps.clearPreview(layer.id),
        controller: deps.controller,
        isDraftValid: (next) => isFilterConfigValid(next.type, next.settings),
        makeDurable: (imageName) => makeDurable(imageName),
        runFilter: (options) =>
          runLayerFilter({
            ...options,
            deps: {
              encodeSurface: deps.encodeSurface,
              runFilterGraph: deps.runFilterGraph,
              uploadIntermediate: deps.uploadIntermediate,
            },
          }),
      },
      guard,
      initialDraft: draft,
      initialFilter,
      layerName: layer.name,
      layerType: layer.type,
    });
    if (!session) {
      return 'not-ready';
    }
    const installed = session;
    unsubscribeSession = installed.subscribe(syncStore);
    unsubscribeController = deps.controller.subscribe(() => {
      if (session === installed && deps.controller.getSnapshot().status === 'idle') {
        clear();
      }
    });
    syncStore();
    deps.setViewTool();
    return 'started';
  };

  return {
    cancel: clear,
    commit: async (target, makeImageDurable) => {
      if (deps.isInteractionLocked()) {
        return 'blocked';
      }
      const owned = session;
      if (!owned) {
        return 'stale';
      }
      makeDurable = makeImageDurable;
      const result = await owned.commit(target);
      if (result === 'committed' && session === owned) {
        clear();
      }
      return result;
    },
    dispose: clear,
    interruptAndBlock: () => {
      session?.interruptProcessing();
      session?.blockCommit();
    },
    isActive: () => session !== null,
    process: async () => {
      if (deps.isInteractionLocked()) {
        return 'blocked';
      }
      if (!session) {
        return 'stale';
      }
      const result: CanvasOperationRunResult = await session.process();
      return result === 'published' ? 'completed' : 'stale';
    },
    reset: (settings) => {
      if (deps.isInteractionLocked()) {
        return 'blocked';
      }
      if (!session) {
        return 'stale';
      }
      session.reset(settings);
      return 'updated';
    },
    setAutoProcess: (value) => {
      if (deps.isInteractionLocked()) {
        return 'blocked';
      }
      if (!session) {
        return 'stale';
      }
      session.setAutoProcess(value);
      return 'updated';
    },
    start,
    updateDraft: (draft) => {
      if (deps.isInteractionLocked()) {
        return 'blocked';
      }
      if (!session) {
        return 'stale';
      }
      session.updateDraft(draft);
      return 'updated';
    },
  };
};

export type { FilterOperationSessionState } from '@workbench/canvas-operations/operationTypes';
