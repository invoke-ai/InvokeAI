import type { CommitStagedImageOptions, CommitStagedImageResult } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV2,
  CanvasRasterLayerContractV2,
  CanvasStagingCandidateContract,
  CanvasStateContractV2,
} from '@workbench/canvas-engine/contracts';
import type { History } from '@workbench/canvas-engine/history/history';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { ProjectEvent } from '@workbench/projectContracts';

import { getCanvasStagingCandidateFingerprint } from '@workbench/canvasStagingView';

export interface StagedResultControllerOptions<Permit, Owner = symbol> {
  readonly capturePermit: (owner?: Owner) => Permit | null;
  readonly createEventId: () => string;
  readonly createLayerId: () => string;
  readonly dispatchPrepared: (
    mutation: CanvasProjectMutation,
    reducerAccepted: () => boolean,
    mirrorAccepted: () => boolean,
    origin?: 'system' | 'user'
  ) => void;
  readonly endBurst: () => void;
  readonly getCanvasState: () => CanvasStateContractV2 | null;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly history: History;
  readonly isGestureActive: () => boolean;
  readonly isPermitCurrent: (permit: Permit) => boolean;
  readonly now: () => string;
}

const createLayer = (
  id: string,
  name: string,
  candidate: CanvasStagingCandidateContract
): CanvasRasterLayerContractV2 => {
  const { placement } = candidate;
  return {
    blendMode: 'normal',
    id,
    isEnabled: true,
    isLocked: false,
    name,
    opacity: placement.opacity,
    source: {
      image: { height: candidate.height, imageName: candidate.imageName, width: candidate.width },
      type: 'image',
    },
    transform: {
      rotation: 0,
      scaleX: candidate.width === 0 ? 1 : placement.width / candidate.width,
      scaleY: candidate.height === 0 ? 1 : placement.height / candidate.height,
      x: placement.x,
      y: placement.y,
    },
    type: 'raster',
  };
};

/** Owns guarded, project-bound acceptance of staged canvas results. */
export class StagedResultController<Permit, Owner = symbol> {
  private disposed = false;

  constructor(private readonly options: StagedResultControllerOptions<Permit, Owner>) {}

  commit(options: CommitStagedImageOptions, owner?: Owner): CommitStagedImageResult {
    const o = this.options;
    if (this.disposed) {
      return { status: 'missing' };
    }
    const permit = o.capturePermit(owner);
    if (!permit || o.isGestureActive()) {
      return { status: 'busy' };
    }
    const canvas = o.getCanvasState();
    if (!canvas) {
      return { status: 'missing' };
    }
    const candidateFingerprint = getCanvasStagingCandidateFingerprint(options.candidate);
    if (
      !canvas.stagingArea.pendingImages.some(
        (pending) => getCanvasStagingCandidateFingerprint(pending) === candidateFingerprint
      )
    ) {
      return { status: 'missing' };
    }
    if (!o.isPermitCurrent(permit) || o.isGestureActive()) {
      return { status: 'busy' };
    }

    const continueStaging = options.continueStaging === true;
    const layer = {
      ...createLayer(o.createLayerId(), `Layer ${canvas.document.layers.length + 1}`, options.candidate),
      isEnabled: !continueStaging,
    };
    const event: ProjectEvent = {
      createdAt: o.now(),
      id: o.createEventId(),
      summary: continueStaging
        ? `Saved ${options.candidate.imageName} as a disabled raster layer while continuing staging`
        : `Accepted ${options.candidate.imageName} into a new raster layer`,
      type: 'canvas-layer-accepted',
    };
    const previousSelectedLayerId = canvas.document.selectedLayerId;
    const previousLayers = canvas.document.layers;
    const acceptedLayers = [layer, ...previousLayers];
    const previousStagingArea = canvas.stagingArea;
    const acceptedSelectedLayerId = continueStaging ? previousSelectedLayerId : layer.id;
    const hasPreviousLayerStack = (document: CanvasDocumentContractV2 | null): boolean =>
      document?.selectedLayerId === previousSelectedLayerId &&
      document.layers.length === previousLayers.length &&
      document.layers.every((current, index) => current === previousLayers[index]);
    const hasAcceptedLayerStack = (document: CanvasDocumentContractV2 | null): boolean =>
      document?.selectedLayerId === acceptedSelectedLayerId &&
      document.layers.length === acceptedLayers.length &&
      document.layers.every((current, index) => current === acceptedLayers[index]);
    const isCommitted = (next: CanvasStateContractV2 | null): boolean =>
      next?.document.selectedLayerId === acceptedSelectedLayerId &&
      next.document.layers.some((current) => current === layer) &&
      (continueStaging
        ? next.stagingArea === previousStagingArea
        : next.stagingArea.pendingImages.length === 0 &&
          next.stagingArea.pendingImageIds.length === 0 &&
          next.stagingArea.selectedImageIndex === 0 &&
          !next.stagingArea.isVisible);
    const isMirrored = (): boolean =>
      o.getDocument()?.selectedLayerId === acceptedSelectedLayerId &&
      o.getDocument()?.layers.some((current) => current === layer) === true;

    try {
      o.endBurst();
      o.dispatchPrepared(
        {
          candidateFingerprint,
          continueStaging,
          event,
          layer,
          selectedImageIndex: options.selectedImageIndex,
          type: 'commitStagedImage',
        },
        () => isCommitted(o.getCanvasState()),
        isMirrored
      );
    } catch {
      if (isCommitted(o.getCanvasState())) {
        o.dispatchPrepared(
          {
            continueStaging,
            event,
            layer,
            selectedLayerId: previousSelectedLayerId,
            stagingArea: previousStagingArea,
            type: 'rollbackStagedImageCommit',
          },
          () =>
            hasPreviousLayerStack(o.getCanvasState()?.document ?? null) &&
            o.getCanvasState()?.stagingArea === previousStagingArea,
          () => hasPreviousLayerStack(o.getDocument()),
          'system'
        );
      }
      return { status: 'stale' };
    }

    const applyLayerStack = (
      mutation: Extract<CanvasProjectMutation, { type: 'applyCanvasLayerStackMutation' }>,
      reducerAccepted: () => boolean,
      mirrorAccepted: () => boolean,
      rollback: Extract<CanvasProjectMutation, { type: 'applyCanvasLayerStackMutation' }>,
      reducerRolledBack: () => boolean,
      mirrorRolledBack: () => boolean
    ): void => {
      try {
        o.dispatchPrepared(mutation, reducerAccepted, mirrorAccepted);
      } catch (error) {
        if (reducerAccepted()) {
          o.dispatchPrepared(rollback, reducerRolledBack, mirrorRolledBack, 'system');
        }
        throw error;
      }
    };
    const addAcceptedLayer: Extract<CanvasProjectMutation, { type: 'applyCanvasLayerStackMutation' }> = {
      add: { index: 0, layers: [layer] },
      enabledUpdates: [],
      selectedLayerId: acceptedSelectedLayerId,
      type: 'applyCanvasLayerStackMutation',
    };
    const removeAcceptedLayer: Extract<CanvasProjectMutation, { type: 'applyCanvasLayerStackMutation' }> = {
      enabledUpdates: [],
      removeIds: [layer.id],
      selectedLayerId: previousSelectedLayerId,
      type: 'applyCanvasLayerStackMutation',
    };
    o.history.push({
      bytes: 256,
      label: continueStaging ? 'Save staged image as disabled layer' : 'Accept staged image',
      redo: () =>
        applyLayerStack(
          addAcceptedLayer,
          () => hasAcceptedLayerStack(o.getCanvasState()?.document ?? null),
          () => hasAcceptedLayerStack(o.getDocument()),
          removeAcceptedLayer,
          () => hasPreviousLayerStack(o.getCanvasState()?.document ?? null),
          () => hasPreviousLayerStack(o.getDocument())
        ),
      replayFailureAtomic: true,
      undo: () =>
        applyLayerStack(
          removeAcceptedLayer,
          () => hasPreviousLayerStack(o.getCanvasState()?.document ?? null),
          () => hasPreviousLayerStack(o.getDocument()),
          addAcceptedLayer,
          () => hasAcceptedLayerStack(o.getCanvasState()?.document ?? null),
          () => hasAcceptedLayerStack(o.getDocument())
        ),
    });
    return { layerId: layer.id, status: 'committed' };
  }

  dispose(): void {
    this.disposed = true;
  }
}
