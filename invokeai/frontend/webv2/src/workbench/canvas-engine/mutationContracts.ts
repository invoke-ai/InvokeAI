/**
 * The canvas document mutation vocabulary.
 *
 * These types live inside `canvas-engine` because the engine and its controllers
 * are their heaviest consumers — 21 engine files import `CanvasProjectMutation`,
 * and every one of them wants only the type, never the reducer. Keeping the
 * declaration in `workbench/canvasProjectMutations.ts` made the engine's own
 * public surface (`capabilities.ts`) import upward from the workbench root,
 * which is the edge that closed the 16-module import cycle.
 *
 * The reducer that interprets these mutations stays in
 * `workbench/canvasProjectMutations.ts` and re-exports these types, so callers
 * outside canvas are unaffected.
 */

import type { ProjectEvent } from '@workbench/projectEventContracts';

import type {
  CanvasAdjustmentsContract,
  CanvasControlAdapterContract,
  CanvasControlLayerContract,
  CanvasDocumentContractV2,
  CanvasLayerBaseContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasMaskContract,
  CanvasRasterLayerContractV2,
  CanvasRegionalGuidanceLayerContract,
  CanvasStagingAreaContractV2,
} from './contracts';

export type CanvasLayerBasePatch = Partial<
  Pick<CanvasLayerBaseContract, 'name' | 'isEnabled' | 'isLocked' | 'opacity' | 'blendMode'>
> & { transform?: Partial<CanvasLayerBaseContract['transform']> };

export type CanvasLayerConfigPatch =
  | {
      layerType: 'raster';
      adjustments?: CanvasAdjustmentsContract;
      isTransparencyLocked?: boolean;
      filter?: CanvasRasterLayerContractV2['filter'];
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
  | { layerType: 'inpaint_mask'; mask?: Partial<CanvasMaskContract>; noiseLevel?: number; denoiseLimit?: number };

export type CanvasProjectMutation =
  | {
      type: 'commitStagedImage';
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
  | { type: 'addCanvasLayer'; layer: CanvasLayerContract; index?: number }
  | {
      type: 'applyCanvasLayerStackMutation';
      add?: { index: number; layers: readonly CanvasLayerContract[] };
      removeIds?: readonly string[];
      enabledUpdates: readonly { id: string; isEnabled: boolean }[];
      lockedUpdates?: readonly { id: string; isLocked: boolean }[];
      orderedIds?: readonly string[];
      /** Omit to preserve the current selection, repairing it if that layer is removed. */
      selectedLayerId?: string | null;
    }
  | { type: 'removeCanvasLayers'; ids: string[] }
  | { type: 'duplicateCanvasLayer'; sourceId: string; newId: string }
  | { type: 'reorderCanvasLayers'; orderedIds: string[] }
  | { type: 'updateCanvasLayer'; id: string; patch: CanvasLayerBasePatch }
  | { type: 'replaceCanvasLayer'; layerId: string; layer: CanvasLayerContract }
  | { type: 'setCanvasLayersEnabled'; updates: readonly { id: string; isEnabled: boolean }[] }
  | { type: 'setCanvasLayerPositions'; updates: readonly { id: string; x: number; y: number }[] }
  | { type: 'setCanvasLayersHidden'; updates: readonly { id: string; isHidden: boolean }[] }
  | { type: 'updateCanvasLayerSource'; id: string; source: CanvasLayerSourceContract }
  | { type: 'updateCanvasLayerConfig'; id: string; config: CanvasLayerConfigPatch }
  | { type: 'convertCanvasLayer'; id: string; targetType: CanvasLayerContract['type']; layer: CanvasLayerContract }
  | {
      type: 'mergeCanvasLayersDown';
      upperLayerId: string;
      source: Extract<CanvasLayerSourceContract, { type: 'paint' }>;
    }
  | { type: 'setCanvasBbox'; bbox: CanvasDocumentContractV2['bbox'] }
  | { type: 'setCanvasSelectedLayer'; id: string | null }
  | { type: 'resizeCanvasDocument'; width: number; height: number; offsetX?: number; offsetY?: number }
  | { type: 'replaceCanvasDocument'; document: CanvasDocumentContractV2 }
  | { type: 'saveCanvasSnapshot'; id: string; name: string; createdAt: string }
  | { type: 'restoreCanvasSnapshot'; snapshotId: string }
  | { type: 'deleteCanvasSnapshot'; snapshotId: string }
  | { type: 'setCanvasStagingAutoSwitch'; mode: CanvasStagingAreaContractV2['autoSwitchMode'] };
