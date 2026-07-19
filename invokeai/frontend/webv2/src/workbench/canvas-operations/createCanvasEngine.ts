import type {
  CanvasEngineImplementation as CoreCanvasEngineImplementation,
  CanvasEngineOptions as CoreCanvasEngineOptions,
} from '@workbench/canvas-engine/engine';
import type { CanvasUtilityGraphResult } from '@workbench/canvas-operations/contracts';
import type { GenerationCompositeHost } from '@workbench/canvas-operations/generationComposite';
import type { BackendGraphContract } from '@workbench/graphContracts';

import { createCanvasEngine as createCanvasEngineCore } from '@workbench/canvas-engine/engine';
import { canvasApplicationPort } from '@workbench/canvas-operations/applicationPort';
import { createBoundedCompositeDedupeCache } from '@workbench/canvas-operations/compositeForGeneration';
import { composeForGeneration } from '@workbench/canvas-operations/generationComposite';

import type { CanvasOperationCapability, CanvasOperationImplementation } from './contracts';

import { attachCanvasOperations } from './operationAccess';

export interface CanvasEngineOptions extends Omit<CoreCanvasEngineOptions, 'uploadImage' | 'getMainModelBase'> {
  getMainModelBase?: () => string | null;
  selectObjectDeps?: {
    uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ height: number; imageName: string; width: number }>;
    runGraph(options: {
      graph: BackendGraphContract;
      outputNodeId?: string;
      signal?: AbortSignal;
    }): Promise<CanvasUtilityGraphResult>;
  };
  filterDeps?: {
    uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ imageName: string }>;
    runGraph(options: {
      graph: BackendGraphContract;
      outputNodeId?: string;
      signal?: AbortSignal;
    }): Promise<CanvasUtilityGraphResult>;
  };
}

export type CanvasOperationsCapability = CanvasOperationCapability;
/** Private application composition shape; public callers receive the capability-only handle from the registry API. */
export type CanvasEngine = CoreCanvasEngineImplementation;

/** Application composition root: owns SAM/filter sessions, queues, uploads, and their core capability adapters. */
export const createCanvasEngine = (options: CanvasEngineOptions): CanvasEngine => {
  const { filterDeps, selectObjectDeps, ...coreOptions } = options;
  const composition = createCanvasEngineCore({
    ...coreOptions,
    getMainModelBase: options.getMainModelBase,
    uploadImage: (blob) => canvasApplicationPort.uploadImage(blob),
  });
  const { applicationHost: host } = composition;
  const core = composition.engine;
  const stores = canvasApplicationPort.createOperationStores();
  const controller = canvasApplicationPort.createOperationController({
    edits: core.edits,
    isGuardCurrent: host.isGuardCurrent,
  });

  let filterCoordinator!: ReturnType<typeof canvasApplicationPort.createFilterCoordinator>;
  const selectObjectCoordinator = canvasApplicationPort.createSelectObjectCoordinator({
    captureGuard: host.captureGuard,
    clearOtherOperation: () => filterCoordinator.cancel(),
    clearPreview: host.clearSamPreview,
    commitGenerated: (preview, commitOptions) =>
      host.commitGenerated({
        copyLayerName:
          commitOptions.mode === 'copy-raster'
            ? 'Segmented Object'
            : commitOptions.mode === 'copy-control'
              ? 'Segmented Object Control'
              : undefined,
        guard: preview.guard,
        historyLabel: commitOptions.mode === 'replace' ? 'Replace layer with selected object' : undefined,
        image: preview.image,
        origin: { x: preview.rect.x, y: preview.rect.y },
        signal: commitOptions.signal,
        target:
          commitOptions.mode === 'replace'
            ? 'replace'
            : commitOptions.mode === 'copy-raster'
              ? 'copy-raster'
              : 'copy-control',
      }),
    commitMask: (preview, target, signal) =>
      host.commitMask({ guard: preview.guard, image: preview.image, rect: preview.rect, signal, target }),
    controller,
    decodePreview: host.decodeSelectObjectPreview,
    exportSource: host.exportBakedLayerBlob,
    invalidateOverlay: () => {
      const session = stores.samSession.get();
      host.setSamInteraction(
        session?.input.type === 'visual'
          ? { input: session.input, pointLabel: session.pointLabel, sourceRect: session.sourceRect }
          : null
      );
    },
    isGuardCurrent: host.isGuardCurrent,
    isInteractionLocked: host.isInteractionLocked,
    isSamToolActive: host.isSamToolActive,
    prepareStart: host.prepareSelectObjectStart,
    projectId: options.projectId,
    publishPreview: (preview) => host.publishSamPreview(preview),
    replaceSelection: (preview, signal) => host.replaceSelection(preview.guard, preview.image, preview.rect, signal),
    replaceTemporaryRestoreTool: host.replaceTemporaryRestoreTool,
    runGraph: (runOptions) => selectObjectDeps?.runGraph(runOptions) ?? canvasApplicationPort.runGraph(runOptions),
    selectLayer: host.selectLayer,
    setSamTool: host.setSamTool,
    setViewTool: host.setViewTool,
    stores,
    uploadIntermediate: async (blob, signal) => {
      if (selectObjectDeps) {
        return selectObjectDeps.uploadIntermediate(blob, signal);
      }
      if (signal?.aborted) {
        throw new DOMException('Select Object upload was aborted.', 'AbortError');
      }
      const uploaded = await canvasApplicationPort.uploadImage(blob, { isIntermediate: true, signal });
      if (signal?.aborted) {
        throw new DOMException('Select Object upload was aborted.', 'AbortError');
      }
      return uploaded;
    },
  });

  filterCoordinator = canvasApplicationPort.createFilterCoordinator({
    captureGuard: host.captureGuard,
    clearOtherOperation: () => selectObjectCoordinator.cancel(),
    clearPreview: host.clearFilterPreview,
    controller,
    encodeSurface: host.encodeSurface,
    getDocument: host.getDocument,
    getSessionDeps: (layerId) => ({
      commit: ({ draft, guard, image, rect, signal, target }) => {
        if (host.isInteractionLocked()) {
          return Promise.resolve({ status: 'locked' });
        }
        return host.commitFilter({
          filter: draft,
          guard,
          image,
          mode: target === 'apply' ? 'replace' : 'copy',
          rect,
          requireExactImageDimensions: true,
          signal,
          target,
        });
      },
      exportPixels: () => host.exportLayerPixels(layerId, { applyAdjustments: true, includeDisabled: true }),
      isGuardCurrent: host.isGuardCurrent,
      publishPreview: (imageName, rect, guard, filterType) =>
        host.publishFilterPreview(guard.layerId, imageName, rect, guard, filterType),
    }),
    isInteractionLocked: host.isInteractionLocked,
    runFilterGraph: async (runOptions) => {
      const output = await (filterDeps?.runGraph(runOptions) ?? canvasApplicationPort.runGraph(runOptions));
      return { height: output.height, imageName: output.imageName, width: output.width };
    },
    selectLayer: host.selectLayer,
    setViewTool: host.setViewTool,
    stores,
    uploadIntermediate: async (blob, signal) => {
      const uploaded = await (filterDeps?.uploadIntermediate(blob, signal) ??
        canvasApplicationPort.uploadImage(blob, { isIntermediate: true, signal }));
      return { imageName: uploaded.imageName };
    },
  });

  const syncEditingLock = (): void =>
    core.stores.documentEditingLocked.set(controller.getSnapshot().status === 'active');
  const unsubscribeEditingLock = controller.subscribe(syncEditingLock);
  const syncSamInteraction = (): void => {
    const session = stores.samSession.get();
    host.setSamInteraction(
      session?.input.type === 'visual'
        ? { input: session.input, pointLabel: session.pointLabel, sourceRect: session.sourceRect }
        : null
    );
  };
  const unsubscribeSamInteraction = stores.samSession.subscribe(syncSamInteraction);
  const unsubscribeToolChanges = host.subscribeToolChanges(({ from, temporary, to }) => {
    if (from === 'sam' && to !== 'sam' && !temporary && selectObjectCoordinator.isActive()) {
      selectObjectCoordinator.cancel();
    }
  });
  host.setSamInputHandler((input) => selectObjectCoordinator.update({ input }));
  host.setEscapeHandler((gestureWasActive) => {
    if (gestureWasActive) {
      return false;
    }
    if (filterCoordinator.isActive()) {
      filterCoordinator.cancel();
      return true;
    }
    if (selectObjectCoordinator.isActive()) {
      selectObjectCoordinator.cancel();
      return true;
    }
    return false;
  });
  syncEditingLock();
  syncSamInteraction();

  const compositeDedupe = createBoundedCompositeDedupeCache();
  const generationCompositeHost: GenerationCompositeHost = {
    captureDocumentSnapshot: () => core.document.captureSnapshot(),
    captureRasterSnapshot: (snapshot, layerIds, captureOptions) =>
      core.exports.captureRasterSnapshot(snapshot, layerIds, captureOptions),
    dedupe: compositeDedupe,
    getCompositeExecutorDeps: () => core.exports.getCompositeExecutorDeps(),
  };

  const operations: CanvasOperationImplementation = {
    applySelectObjectSession: selectObjectCoordinator.apply,
    cancelFilterOperation: filterCoordinator.cancel,
    cancelSelectObjectSession: selectObjectCoordinator.cancel,
    commitFilterOperation: filterCoordinator.commit,
    composeForGeneration: (composeOptions) => composeForGeneration(generationCompositeHost, composeOptions),
    controller,
    getFilterSessionState: stores.filterSession.get,
    getOperationState: controller.getSnapshot,
    getSamSessionState: stores.samSession.get,
    processFilterOperation: filterCoordinator.process,
    processSelectObjectSession: selectObjectCoordinator.process,
    resetFilterOperation: filterCoordinator.reset,
    resetSelectObjectSession: selectObjectCoordinator.reset,
    saveSelectObjectSession: selectObjectCoordinator.save,
    setFilterOperationAutoProcess: filterCoordinator.setAutoProcess,
    startFilterOperation: filterCoordinator.start,
    startSelectObject: selectObjectCoordinator.start,
    stores,
    subscribeFilterSession: stores.filterSession.subscribe,
    subscribeOperation: controller.subscribe,
    subscribeSamSession: stores.samSession.subscribe,
    updateFilterOperation: filterCoordinator.updateDraft,
    updateSelectObjectSession: selectObjectCoordinator.update,
    uploadIntermediate: async (blob, signal) => {
      if (signal?.aborted) {
        throw new DOMException('Canvas upload aborted', 'AbortError');
      }
      const uploaded = await canvasApplicationPort.uploadImage(blob, { isIntermediate: true });
      if (signal?.aborted) {
        throw new DOMException('Canvas upload aborted', 'AbortError');
      }
      return { imageName: uploaded.imageName };
    },
  };

  const coreSetInteractionLocked = core.tools.setInteractionLocked;
  const tools = {
    ...core.tools,
    setInteractionLocked: (locked: boolean) => {
      if (locked) {
        selectObjectCoordinator.interruptAndBlock();
        filterCoordinator.interruptAndBlock();
      }
      coreSetInteractionLocked(locked);
      if (!locked && selectObjectCoordinator.isActive()) {
        host.setSamTool();
      }
    },
  };

  let disposed = false;
  const disposeOperations = (): void => {
    if (disposed) {
      return;
    }
    disposed = true;
    host.setSamInputHandler(null);
    host.setEscapeHandler(null);
    unsubscribeToolChanges();
    unsubscribeSamInteraction();
    unsubscribeEditingLock();
    filterCoordinator.dispose();
    selectObjectCoordinator.dispose();
    controller.dispose();
    core.stores.documentEditingLocked.set(false);
    host.setSamInteraction(null);
  };
  const lifecycle = {
    ...core.lifecycle,
    beginCooldown: () => {
      filterCoordinator.cancel();
      selectObjectCoordinator.cancel();
      controller.invalidateDocument(options.projectId);
      return core.lifecycle.beginCooldown();
    },
    dispose: () => {
      disposeOperations();
      core.lifecycle.dispose();
    },
  };
  const diagnostics = {
    ...core.diagnostics,
    clearCaches: async () => {
      filterCoordinator.cancel();
      selectObjectCoordinator.cancel();
      await core.diagnostics.clearCaches();
    },
  };

  const engine: CanvasEngine = { ...core, diagnostics, lifecycle, tools };
  attachCanvasOperations(engine, operations);
  return engine;
};
