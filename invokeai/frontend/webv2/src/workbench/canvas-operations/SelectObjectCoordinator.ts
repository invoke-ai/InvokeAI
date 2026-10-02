import type { CommitGeneratedImageResult } from '@workbench/canvas-engine/api';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { SaveSelectObjectSessionResult, SelectObjectSaveTarget } from '@workbench/canvas-operations/api';
import type {
  CanvasSelectObjectCoordinatorDeps,
  CanvasSelectObjectOperationCoordinator,
  SelectObjectSession,
  SelectObjectSessionPreview,
} from '@workbench/canvas-operations/contracts';

import { createSelectObjectSession } from './selectObjectSession';

interface CommitOwner {
  controller: AbortController;
  inputHash: string;
  preview: SelectObjectSessionPreview<RasterSurface>;
  previewId: number;
  session: SelectObjectSession<RasterSurface>;
  token: number;
}

export const createSelectObjectOperationCoordinator = (
  deps: CanvasSelectObjectCoordinatorDeps
): CanvasSelectObjectOperationCoordinator => {
  let session: SelectObjectSession<RasterSurface> | null = null;
  let unsubscribe: (() => void) | null = null;
  let controllerUnsubscribe: (() => void) | null = null;
  let startContext: Extract<ReturnType<typeof deps.prepareStart>, { status: 'ready' }> | null = null;
  let publishedGuard: SelectObjectSessionPreview<RasterSurface>['guard'] | null = null;
  let pointLabel: 'include' | 'exclude' = 'include';
  let commitOwner: CommitOwner | null = null;
  let commitToken = 0;

  const syncStore = (): void => {
    if (!session || !startContext) {
      deps.stores.samSession.set(null);
      return;
    }
    const state = session.getSnapshot();
    deps.stores.samSession.set({
      applyPolygonRefinement: state.applyPolygonRefinement,
      autoProcess: state.autoProcess,
      error: state.error,
      hasPreview: state.preview !== null,
      layerName: startContext.layerName,
      layerType: startContext.layerType,
      input:
        state.input.type === 'visual'
          ? {
              bbox: state.input.bbox ? { ...state.input.bbox } : null,
              excludePoints: state.input.excludePoints.map((point) => ({ ...point })),
              includePoints: state.input.includePoints.map((point) => ({ ...point })),
              type: 'visual',
            }
          : { prompt: state.input.prompt, type: 'prompt' },
      invert: state.invert,
      isolatedPreview: state.isolatedPreview,
      model: state.model,
      pointLabel,
      sourceRect: startContext.sourceRect,
      status: commitOwner?.session === session ? 'committing' : state.status,
    });
    deps.invalidateOverlay();
  };

  const revokeCommit = (): void => {
    commitToken += 1;
    const owner = commitOwner;
    commitOwner = null;
    owner?.controller.abort();
  };

  const invalidateCommit = (): void => {
    revokeCommit();
    syncStore();
  };

  const isOwnerCurrent = (owner: CommitOwner): boolean => {
    const state = owner.session.getSnapshot();
    return (
      commitOwner === owner &&
      commitToken === owner.token &&
      session === owner.session &&
      publishedGuard === owner.preview.guard &&
      state.preview?.previewId === owner.previewId &&
      state.preview.inputHash === owner.inputHash &&
      deps.isGuardCurrent(owner.preview.guard) &&
      !owner.controller.signal.aborted
    );
  };

  const beginCommit = (
    currentSession: SelectObjectSession<RasterSurface>,
    preview: SelectObjectSessionPreview<RasterSurface>
  ): CommitOwner | null => {
    if (commitOwner) {
      return null;
    }
    const owner: CommitOwner = {
      controller: new AbortController(),
      inputHash: preview.inputHash,
      preview,
      previewId: preview.previewId,
      session: currentSession,
      token: ++commitToken,
    };
    commitOwner = owner;
    syncStore();
    return owner;
  };

  const finishCommit = (owner: CommitOwner): boolean => {
    if (!isOwnerCurrent(owner)) {
      return false;
    }
    commitOwner = null;
    syncStore();
    return true;
  };

  const clear = (expectedOwner?: CommitOwner): boolean => {
    if (expectedOwner && (!isOwnerCurrent(expectedOwner) || session !== expectedOwner.session)) {
      return false;
    }
    deps.replaceTemporaryRestoreTool();
    const previous = session;
    session = null;
    unsubscribe?.();
    unsubscribe = null;
    controllerUnsubscribe?.();
    controllerUnsubscribe = null;
    startContext = null;
    publishedGuard = null;
    pointLabel = 'include';
    revokeCommit();
    deps.clearPreview();
    deps.controller.cancel();
    previous?.dispose();
    deps.stores.samSession.set(null);
    deps.invalidateOverlay();
    return true;
  };

  const currentPreview = (): SelectObjectSessionPreview<RasterSurface> | null => {
    const state = session?.getSnapshot();
    const preview = state?.preview;
    if (
      !state ||
      (state.status !== 'ready' && state.status !== 'error') ||
      !preview ||
      preview.guard !== publishedGuard
    ) {
      return null;
    }
    return deps.isGuardCurrent(preview.guard) ? preview : null;
  };

  const reportResultError = (owner: CommitOwner, result: SaveSelectObjectSessionResult): void => {
    if (!finishCommit(owner)) {
      return;
    }
    owner.session.reportError(
      result.status === 'locked'
        ? { code: 'locked' }
        : {
            code: 'unknown',
            detail: result.status === 'failed' ? result.message : `Select Object commit is ${result.status}.`,
          }
    );
  };

  const makeDurable = async (
    owner: CommitOwner,
    callback: (imageName: string) => Promise<void>
  ): Promise<{ status: 'ok' } | { status: 'failed'; message: string }> => {
    try {
      await callback(owner.preview.image.imageName);
      return { status: 'ok' };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (finishCommit(owner)) {
        owner.session.reportError({ code: 'unknown', detail: message });
      }
      return { message, status: 'failed' };
    }
  };

  return {
    start: (layerId) => {
      if (deps.isInteractionLocked()) {
        return 'locked';
      }
      const prepared = deps.prepareStart(layerId);
      if (prepared.status !== 'ready') {
        return prepared.status;
      }
      deps.clearOtherOperation();
      clear();
      startContext = prepared;
      deps.selectLayer(layerId);
      const operation = deps.controller.start({
        cleanupPreview: deps.clearPreview,
        guard: prepared.guard,
        identity: { kind: 'select-object', layerId, projectId: deps.projectId },
      });
      if (!operation) {
        startContext = null;
        return 'not-ready';
      }
      const createSession = deps.createSession ?? createSelectObjectSession;
      session = createSession({
        deps: {
          captureGuard: () => deps.captureGuard(layerId),
          cleanupPreview: deps.clearPreview,
          controller: deps.controller,
          decodePreview: deps.decodePreview,
          exportSource: () => deps.exportSource(layerId),
          isGuardCurrent: deps.isGuardCurrent,
          publishPreview: (preview) => {
            publishedGuard = preview.guard;
            deps.publishPreview(preview);
            return undefined;
          },
          runGraph: deps.runGraph,
          uploadIntermediate: deps.uploadIntermediate,
        },
        operation,
      });
      const installed = session;
      unsubscribe = installed.subscribe(() => {
        if (
          commitOwner?.session === installed &&
          installed.getSnapshot().preview?.previewId !== commitOwner.previewId
        ) {
          invalidateCommit();
        }
        syncStore();
      });
      controllerUnsubscribe = deps.controller.subscribe(() => {
        if (session === installed && deps.controller.getSnapshot().status === 'idle') {
          clear();
          if (deps.isSamToolActive()) {
            deps.setViewTool();
          }
        }
      });
      installed.update({ input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' } });
      syncStore();
      deps.setSamTool();
      return 'started';
    },
    update: (changes) => {
      if (deps.isInteractionLocked()) {
        return 'blocked';
      }
      if (!session) {
        return 'stale';
      }
      const { pointLabel: nextPointLabel, ...sessionChanges } = changes;
      const state = session.getSnapshot();
      const processingChanged =
        ('input' in sessionChanges && sessionChanges.input !== state.input) ||
        ('model' in sessionChanges && sessionChanges.model !== state.model) ||
        ('invert' in sessionChanges && sessionChanges.invert !== state.invert) ||
        ('applyPolygonRefinement' in sessionChanges &&
          sessionChanges.applyPolygonRefinement !== state.applyPolygonRefinement);
      if (processingChanged) {
        invalidateCommit();
      }
      if (nextPointLabel) {
        pointLabel = nextPointLabel;
      }
      session.update(sessionChanges);
      syncStore();
      return 'updated';
    },
    process: () => {
      if (deps.isInteractionLocked()) {
        return Promise.resolve('blocked');
      }
      invalidateCommit();
      return session?.process() ?? Promise.resolve('stale');
    },
    apply: async (durable): Promise<CommitGeneratedImageResult> => {
      if (deps.isInteractionLocked()) {
        return { status: 'locked' };
      }
      const currentSession = session;
      const preview = currentPreview();
      if (!currentSession || !preview) {
        return { status: 'stale' };
      }
      const owner = beginCommit(currentSession, preview);
      if (!owner) {
        return { status: 'busy' };
      }
      if (!isOwnerCurrent(owner)) {
        invalidateCommit();
        return { status: 'stale' };
      }
      const durability = await makeDurable(owner, durable);
      if (durability.status === 'failed') {
        return durability;
      }
      if (!isOwnerCurrent(owner)) {
        return deps.isInteractionLocked() ? { status: 'locked' } : { status: 'stale' };
      }
      if (deps.isInteractionLocked()) {
        invalidateCommit();
        return { status: 'locked' };
      }
      let result: CommitGeneratedImageResult;
      try {
        result = await deps.commitGenerated(preview, { mode: 'replace', signal: owner.controller.signal });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (finishCommit(owner)) {
          owner.session.reportError({ code: 'unknown', detail: message });
        }
        return { message, status: 'failed' };
      }
      if (!isOwnerCurrent(owner)) {
        return result;
      }
      if (result.status === 'committed') {
        clear(owner);
        if (deps.isSamToolActive()) {
          deps.setViewTool();
        }
      } else {
        reportResultError(owner, result);
      }
      return result;
    },
    save: async (target: SelectObjectSaveTarget, durable): Promise<SaveSelectObjectSessionResult> => {
      if (deps.isInteractionLocked()) {
        return { status: 'locked' };
      }
      const currentSession = session;
      const preview = currentPreview();
      if (!currentSession || !preview) {
        return { status: 'stale' };
      }
      const owner = beginCommit(currentSession, preview);
      if (!owner) {
        return { status: 'busy' };
      }
      if (target === 'selection') {
        const result = await deps.replaceSelection(preview, owner.controller.signal);
        if (!isOwnerCurrent(owner)) {
          return result;
        }
        if (result.status === 'selected') {
          clear(owner);
          if (deps.isSamToolActive()) {
            deps.setViewTool();
          }
        } else {
          reportResultError(owner, result);
        }
        return result;
      }
      const durability = await makeDurable(owner, durable);
      if (durability.status === 'failed') {
        return durability;
      }
      if (!isOwnerCurrent(owner)) {
        return deps.isInteractionLocked() ? { status: 'locked' } : { status: 'stale' };
      }
      if (deps.isInteractionLocked()) {
        invalidateCommit();
        return { status: 'locked' };
      }
      let result: SaveSelectObjectSessionResult;
      try {
        result =
          target === 'raster' || target === 'control'
            ? await deps.commitGenerated(preview, {
                mode: target === 'raster' ? 'copy-raster' : 'copy-control',
                signal: owner.controller.signal,
              })
            : await deps.commitMask(preview, target, owner.controller.signal);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (finishCommit(owner)) {
          owner.session.reportError({ code: 'unknown', detail: message });
        }
        return { message, status: 'failed' };
      }
      if (!isOwnerCurrent(owner)) {
        return result;
      }
      if (result.status === 'committed') {
        finishCommit(owner);
      } else {
        reportResultError(owner, result);
      }
      return result;
    },
    reset: () => {
      if (deps.isInteractionLocked()) {
        return 'blocked';
      }
      const operation = deps.controller.getSnapshot();
      if (
        !session ||
        !startContext ||
        !deps.isGuardCurrent(startContext.guard) ||
        operation.status !== 'active' ||
        operation.identity.kind !== 'select-object' ||
        operation.identity.projectId !== deps.projectId ||
        operation.identity.layerId !== startContext.layerId
      ) {
        return 'stale';
      }
      invalidateCommit();
      publishedGuard = null;
      pointLabel = 'include';
      session.reset();
      session.update({ input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' } });
      syncStore();
      return 'updated';
    },
    cancel: () => {
      clear();
      if (deps.isSamToolActive()) {
        deps.setViewTool();
      }
    },
    interruptAndBlock: () => {
      invalidateCommit();
      session?.interruptProcessing();
    },
    isActive: () => session !== null,
    dispose: () => clear(),
  };
};
