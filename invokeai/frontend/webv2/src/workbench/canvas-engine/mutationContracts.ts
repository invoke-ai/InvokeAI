import type { ProjectEvent } from '@workbench/projectEventContracts';

import type {
  CanvasLayerRegionContract,
  CanvasAdjustmentsContract,
  CanvasControlAdapterContract,
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasLayerBaseContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasMaskContract,
  CanvasMaskDenoiseContract,
  CanvasMaskNoiseContract,
  CanvasRasterLayerContractV2,
  CanvasRegionalGuidanceLayerContract,
  CanvasStagingAreaContractV2,
} from './contracts';
import type { CanvasNodeInsertion, CanvasNodeInsertionAnchor, CanvasNodeMove } from './document/insertionAnchors';
import type { ReorderSiblingsCommand } from './document/layerStacks';

/** Base fields a node accepts; a group takes everything but `transform`. */
export type CanvasLayerBasePatch = Partial<
  Pick<CanvasLayerBaseContract, 'name' | 'isEnabled' | 'isLocked' | 'opacity' | 'blendMode' | 'colorLabel'>
> & { transform?: Partial<CanvasLayerBaseContract['transform']> };

/** The base fields a group carries; anything else in a patch does not apply to it. */
export const GROUP_PATCH_KEYS: readonly (keyof CanvasLayerBasePatch)[] = [
  'name',
  'isEnabled',
  'isLocked',
  'opacity',
  'blendMode',
  'colorLabel',
];

/** Patch keys allowed on a locked node (organizational edits never need the lock lifted first). */
export const LOCK_EXEMPT_PATCH_KEYS: readonly (keyof CanvasLayerBasePatch)[] = [
  'name',
  'isEnabled',
  'isLocked',
  'colorLabel',
];

export type CanvasLayerConfigPatch =
  | {
      layerType: 'raster';
      adjustments?: CanvasAdjustmentsContract;
      /** `null` removes the region; absent leaves it untouched. */
      inpaint?: CanvasLayerRegionContract | null;
      isTransparencyLocked?: boolean;
      filter?: CanvasRasterLayerContractV2['filter'];
    }
  | {
      /** Raster-stack groups only; the model refuses the patch elsewhere. */
      layerType: 'group';
      adjustments?: CanvasAdjustmentsContract;
    }
  | {
      layerType: 'control';
      adapter?: Partial<CanvasControlAdapterContract>;
      withTransparencyEffect?: boolean;
      filter?: CanvasControlLayerContract['filter'];
    }
  | {
      layerType: 'regional_guidance';
      mask?: Partial<CanvasMaskContract>;
      positivePrompt?: string | null;
      negativePrompt?: string | null;
      autoNegative?: boolean;
      referenceImages?: CanvasRegionalGuidanceLayerContract['referenceImages'];
    }
  | {
      layerType: 'inpaint_mask';
      mask?: Partial<CanvasMaskContract>;
      /** `null` removes the modifier; absent leaves it untouched. */
      noise?: CanvasMaskNoiseContract | null;
      denoise?: CanvasMaskDenoiseContract | null;
    };

export type CanvasProjectMutation =
  | {
      type: 'commitStagedImage';
      anchor: CanvasNodeInsertionAnchor;
      candidateFingerprint: string;
      continueStaging: boolean;
      event: ProjectEvent;
      layer: CanvasRasterLayerContractV2;
      selectedImageIndex: number;
    }
  | {
      type: 'rollbackStagedImageCommit';
      event: ProjectEvent;
      continueStaging: boolean;
      layer: CanvasRasterLayerContractV2;
      selectedLayerId: string | null;
      stagingArea: CanvasStagingAreaContractV2;
    }
  | { type: 'setStagedImageIndex'; imageIndex: number }
  | { type: 'cycleStagedImage'; direction: -1 | 1 }
  | { type: 'discardSelectedStagedImage' }
  | { type: 'discardAllStagedImages' }
  | { type: 'toggleCanvasStagingVisibility' }
  | { type: 'toggleCanvasStagingThumbnailsVisibility' }
  | { type: 'clearCanvasStaging' }
  | { type: 'addCanvasLayer'; layer: CanvasLayerContract; anchor: CanvasNodeInsertionAnchor }
  | {
      /**
       * One atomic restructuring, applied as `add`, then `move`, then `removeIds`, then the flag
       * and selection updates, so an anchor may name a node a later step moves or removes. The
       * whole mutation is refused when any step is invalid or the result exceeds the depth or
       * node limits.
       */
      type: 'applyCanvasLayerStackMutation';
      add?: readonly CanvasNodeInsertion[];
      move?: readonly CanvasNodeMove[];
      removeIds?: readonly string[];
      enabledUpdates: readonly { id: string; isEnabled: boolean }[];
      lockedUpdates?: readonly { id: string; isLocked: boolean }[];
      /** Omit to preserve the current selection, repairing it if that node is removed. */
      selectedLayerId?: string | null;
    }
  | { type: 'removeCanvasLayers'; ids: string[] }
  | { type: 'reorderCanvasSiblings'; orders: readonly ReorderSiblingsCommand[] }
  | { type: 'updateCanvasLayer'; id: string; patch: CanvasLayerBasePatch }
  | { type: 'replaceCanvasLayer'; layerId: string; layer: CanvasLayerContract }
  | { type: 'setCanvasLayersEnabled'; updates: readonly { id: string; isEnabled: boolean }[] }
  | { type: 'setCanvasLayerPositions'; updates: readonly { id: string; x: number; y: number }[] }
  | { type: 'setCanvasLayersHidden'; updates: readonly { id: string; isHidden: boolean }[] }
  | { type: 'updateCanvasLayerSource'; id: string; source: CanvasLayerSourceContract }
  | { type: 'updateCanvasLayerConfig'; id: string; config: CanvasLayerConfigPatch }
  | {
      /** One atomic config patch across several layers, e.g. moving a modifier between two of them. */
      type: 'updateCanvasLayerConfigs';
      updates: readonly { id: string; config: CanvasLayerConfigPatch }[];
    }
  | {
      type: 'convertCanvasLayer';
      id: string;
      targetType: CanvasLayerContract['type'];
      layer: CanvasLayerContract;
      /** Where a leaf changing stacks lands; the top of its new stack when absent. */
      anchor?: CanvasNodeInsertionAnchor;
    }
  | {
      type: 'mergeCanvasLayersDown';
      upperLayerId: string;
      source: Extract<CanvasLayerSourceContract, { type: 'paint' }>;
    }
  | { type: 'setCanvasBbox'; bbox: CanvasDocumentContractV3['bbox'] }
  | { type: 'setCanvasSelectedLayer'; id: string | null }
  | { type: 'resizeCanvasDocument'; width: number; height: number; offsetX?: number; offsetY?: number }
  /** Replaces text sources while preserving the document revision for one undoable font recovery edit. */
  | { type: 'replaceCanvasFontReferences'; document: CanvasDocumentContractV3 }
  | { type: 'replaceCanvasDocument'; document: CanvasDocumentContractV3 }
  | { type: 'saveCanvasSnapshot'; id: string; name: string; createdAt: string }
  | { type: 'restoreCanvasSnapshot'; snapshotId: string }
  | { type: 'deleteCanvasSnapshot'; snapshotId: string }
  | { type: 'setCanvasStagingAutoSwitch'; mode: CanvasStagingAreaContractV2['autoSwitchMode'] };

/** Why a canvas mutation is dispatched; system work never triggers user-routing policy. */
export type CanvasMutationOrigin = 'user' | 'system';

/** A completed canvas edit that may update aggregate invocation routing. */
export type CanvasEditIntent = { kind: 'paint' } | { kind: 'mutation'; mutation: CanvasProjectMutation };
