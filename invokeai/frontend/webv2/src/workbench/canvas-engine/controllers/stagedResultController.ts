import type { CommitStagedImageOptions, CommitStagedImageResult } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV3,
  CanvasRasterLayerContractV2,
  CanvasStagingCandidateContract,
  CanvasStateContractV3,
} from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { ProjectEvent } from '@workbench/projectContracts';

import { getDocumentLayer, getDocumentLeaves, hasDocumentNode } from '@workbench/canvas-engine/document/documentIndex';
import { insertNodesAtAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import { haveSameStructure } from '@workbench/canvas-engine/document/layerStacks';
import { selectionAfterAcceptedResult } from '@workbench/canvas-engine/document/selectionRepair';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { getCanvasStagingCandidateFingerprint } from '@workbench/canvasStagingView';

import type { CanvasMutationContext, EditStep } from './mutationContext';

import { guardedResultRefusal, layerEditRefusal } from './editSteps';

export interface StagedResultControllerOptions {
  readonly ctx: Pick<
    CanvasMutationContext,
    | 'applyStep'
    | 'begin'
    | 'capturePermit'
    | 'captureInsertionAnchor'
    | 'createLayerId'
    | 'isGestureActive'
    | 'isPermitCurrent'
  >;
  readonly createEventId: () => string;
  readonly getCanvasState: () => CanvasStateContractV3 | null;
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
export class StagedResultController {
  private disposed = false;

  constructor(private readonly options: StagedResultControllerOptions) {}

  commit(options: CommitStagedImageOptions, owner?: symbol): CommitStagedImageResult {
    const o = this.options;
    if (this.disposed) {
      return { status: 'missing' };
    }
    const permit = o.ctx.capturePermit(owner);
    if (!permit || o.ctx.isGestureActive()) {
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
    if (!o.ctx.isPermitCurrent(permit) || o.ctx.isGestureActive()) {
      return { status: 'busy' };
    }

    const continueStaging = options.continueStaging === true;
    const layer = {
      ...createLayer(
        o.ctx.createLayerId(),
        `Layer ${getDocumentLeaves(canvas.document).length + 1}`,
        options.candidate
      ),
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
    const previousStacks = canvas.document.stacks;
    const anchor = o.ctx.captureInsertionAnchor('raster', null);
    const acceptedStacks = insertNodesAtAnchor(previousStacks, anchor, [layer]);
    const previousStagingArea = canvas.stagingArea;
    const acceptedSelectedLayerId = continueStaging
      ? previousSelectedLayerId
      : selectionAfterAcceptedResult(previousStacks, previousSelectedLayerId, layer.id);
    const hasPreviousLayerStack = (document: CanvasDocumentContractV3 | null): boolean =>
      document?.selectedLayerId === previousSelectedLayerId &&
      !hasDocumentNode(document, layer.id) &&
      haveSameStructure(document.stacks, previousStacks);
    const hasAcceptedLayerStack = (document: CanvasDocumentContractV3 | null): boolean =>
      document?.selectedLayerId === acceptedSelectedLayerId &&
      getDocumentLayer(document, layer.id) === layer &&
      haveSameStructure(document.stacks, acceptedStacks);
    const isCommitted = (next: CanvasStateContractV3 | null): boolean =>
      next?.document.selectedLayerId === acceptedSelectedLayerId &&
      getDocumentLayer(next.document, layer.id) === layer &&
      (continueStaging
        ? next.stagingArea === previousStagingArea
        : next.stagingArea.pendingImages.length === 0 &&
          next.stagingArea.pendingImageIds.length === 0 &&
          next.stagingArea.selectedImageIndex === 0 &&
          !next.stagingArea.isVisible);
    type StackMutation = Extract<CanvasProjectMutation, { type: 'applyCanvasLayerStackMutation' }>;
    const addAcceptedLayer: StackMutation = {
      add: [{ anchor, nodes: [layer] }],
      enabledUpdates: [],
      selectedLayerId: acceptedSelectedLayerId,
      type: 'applyCanvasLayerStackMutation',
    };
    const removeAcceptedLayer: StackMutation = {
      enabledUpdates: [],
      removeIds: [layer.id],
      selectedLayerId: previousSelectedLayerId,
      type: 'applyCanvasLayerStackMutation',
    };
    const added: EditStep = {
      accepted: hasAcceptedLayerStack,
      mutation: addAcceptedLayer,
      rollback: { mutation: removeAcceptedLayer, restored: hasPreviousLayerStack },
    };
    const removed: EditStep = {
      accepted: hasPreviousLayerStack,
      mutation: removeAcceptedLayer,
      rollback: { mutation: addAcceptedLayer, restored: hasAcceptedLayerStack },
    };

    // A failed read cannot prove that the commit (or its rollback) landed.
    const readCanvas = (): CanvasStateContractV3 | null => {
      try {
        return o.getCanvasState();
      } catch {
        return null;
      }
    };
    const txn = o.ctx.begin({ historyBytes: HISTORY_ENTRY_OVERHEAD_BYTES, owner });
    if (!('publish' in txn)) {
      return { status: layerEditRefusal(txn.status) };
    }
    try {
      const result = txn.publish(
        continueStaging ? 'Save staged image as disabled layer' : 'Accept staged image',
        {
          accepted: () => isCommitted(readCanvas()),
          mutation: {
            anchor,
            candidateFingerprint,
            continueStaging,
            event,
            layer,
            selectedImageIndex: options.selectedImageIndex,
            type: 'commitStagedImage',
          },
          rollback: {
            mutation: {
              continueStaging,
              event,
              layer,
              selectedLayerId: previousSelectedLayerId,
              stagingArea: previousStagingArea,
              type: 'rollbackStagedImageCommit',
            },
            restored: () => {
              const restored = readCanvas();
              return hasPreviousLayerStack(restored?.document ?? null) && restored?.stagingArea === previousStagingArea;
            },
          },
        },
        {
          bytes: HISTORY_ENTRY_OVERHEAD_BYTES,
          heldAssetRefs: collectHistoryMediaRefs(layer),
          redo: () => o.ctx.applyStep(added),
          undo: () => o.ctx.applyStep(removed),
        }
      );
      return result.status === 'committed'
        ? { layerId: layer.id, status: 'committed' }
        : { status: guardedResultRefusal(result) };
    } finally {
      txn.end();
    }
  }

  dispose(): void {
    this.disposed = true;
  }
}
