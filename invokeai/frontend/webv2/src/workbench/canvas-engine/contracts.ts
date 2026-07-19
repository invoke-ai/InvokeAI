import type { GeneratedImageContract } from '@features/gallery';

export type RegionalGuidanceReferenceImageAsset = GeneratedImageContract;

export type RegionalGuidanceIPAdapterMethod = 'full' | 'style' | 'composition' | 'style_strong' | 'style_precise';

export interface RegionalGuidanceModelRef {
  key: string;
  name: string;
  base: string;
  type: string;
  format?: string;
  variant?: string | null;
  hash?: string;
  submodel_type?: string;
  [key: string]: unknown;
}

export type RegionalGuidanceReferenceImageConfig =
  | {
      type: 'ip_adapter';
      image: RegionalGuidanceReferenceImageAsset | null;
      model: RegionalGuidanceModelRef | null;
      weight: number;
      beginEndStepPct: [number, number];
      method: RegionalGuidanceIPAdapterMethod;
      clipVisionModel: 'ViT-H' | 'ViT-G' | 'ViT-L';
    }
  | {
      type: 'flux_redux';
      image: RegionalGuidanceReferenceImageAsset | null;
      model: RegionalGuidanceModelRef | null;
      imageInfluence: 'lowest' | 'low' | 'medium' | 'high' | 'highest';
    };

export interface RegionalGuidanceReferenceImage {
  id: string;
  isEnabled: boolean;
  config: RegionalGuidanceReferenceImageConfig;
}

export interface CanvasPlacementContract {
  x: number;
  y: number;
  width: number;
  height: number;
  opacity: number;
}

export interface CanvasStagingCandidateContract extends GeneratedImageContract {
  placement: CanvasPlacementContract;
  sourceBackendItemId?: number;
}

export interface CanvasRasterLayerContract extends GeneratedImageContract {
  id: string;
  acceptedAt: string;
  label: string;
  placement: CanvasPlacementContract;
}

export interface CanvasDocumentContract {
  version: 1;
  width: number;
  height: number;
  layers: CanvasRasterLayerContract[];
}

/** @deprecated legacy v1 shape, superseded by {@link CanvasStagingAreaContractV2}. Extracted so v1 and v2 canvas state can share its structure. */
export interface CanvasStagingAreaContract {
  sourceQueueItemId?: string;
  selectedLayerId?: string;
  pendingImageIds: string[];
  pendingImages: CanvasStagingCandidateContract[];
  selectedImageIndex: number;
  isVisible: boolean;
  areThumbnailsVisible: boolean;
}

/** @deprecated legacy v1 shape, superseded by {@link CanvasStateContractV2}. Kept for migration input typing and the still-v1-shaped staging area. */
export interface CanvasStateContract {
  version: 1;
  document: CanvasDocumentContract;
  stagingArea: CanvasStagingAreaContract;
}

// ---------------------------------------------------------------------------
// Canvas v2 document contracts
//
// The canvas document is being rebuilt as a full photo editor: a layer stack
// supporting raster, control, regional-guidance, and inpaint-mask layers, each
// with its own transform/opacity/blend-mode. `CanvasStateContract` (v1) only
// ever produced single-image raster layers positioned by `placement`; v2
// layers are positioned by `transform` and reference bitmaps by `imageName`
// (via `CanvasImageRef`) rather than by resolved URLs, since URLs are
// ephemeral/backend-derived while the document is persisted. The staging
// area (pending queue results awaiting placement onto the canvas) keeps its
// v1 shape unchanged — only the accepted document evolves.
// ---------------------------------------------------------------------------

export type CanvasBlendMode =
  | 'normal'
  | 'multiply'
  | 'screen'
  | 'overlay'
  | 'darken'
  | 'lighten'
  | 'color-dodge'
  | 'color-burn'
  | 'hard-light'
  | 'soft-light'
  | 'difference'
  | 'exclusion'
  | 'hue'
  | 'saturation'
  | 'color'
  | 'luminosity';

/** A reference to a persisted image asset by name, not by resolved URL. */
export interface CanvasImageRef {
  imageName: string;
  width: number;
  height: number;
  contentHash?: string;
}

export type CanvasLayerSourceContract =
  | {
      type: 'paint';
      bitmap: CanvasImageRef | null;
      /**
       * The layer-local origin of `bitmap`'s top-left pixel. Paint layers are
       * content-sized: the persisted bitmap covers only the painted region, and
       * this offset records where that region sits in the layer's local space
       * (it can be negative). Absent (or `{ x: 0, y: 0 }`) for legacy documents
       * whose paint bitmaps were document-sized at the origin — they load
       * identically.
       */
      offset?: { x: number; y: number };
    }
  | { type: 'image'; image: CanvasImageRef }
  | {
      type: 'text';
      content: string;
      fontFamily: string;
      fontSize: number;
      fontWeight: number;
      lineHeight: number;
      align: 'left' | 'center' | 'right';
      color: string;
    }
  | {
      type: 'shape';
      kind: 'rect' | 'ellipse' | 'polygon';
      points?: { x: number; y: number }[];
      width: number;
      height: number;
      fill: string | null;
      stroke: string | null;
      strokeWidth: number;
    }
  | {
      type: 'gradient';
      kind: 'linear' | 'radial';
      angle: number;
      stops: { offset: number; color: string }[];
      /**
       * The gradient's explicit content extent (layer-local pixels). Gradient
       * layers are content-sized like every other layer: the extent is set at
       * creation (bbox-sized) and preserved across angle edits. Absent for legacy
       * documents whose gradients were document-sized by construction — they
       * default to the document dimensions on load (see `getSourceContentRect`).
       */
      width?: number;
      height?: number;
    };

export interface CanvasLayerBaseContract {
  id: string;
  name: string;
  isEnabled: boolean;
  isLocked: boolean;
  opacity: number;
  blendMode: CanvasBlendMode;
  transform: { x: number; y: number; scaleX: number; scaleY: number; rotation: number };
}

export interface CanvasAdjustmentsContract {
  brightness: number;
  contrast: number;
  saturation: number;
  curves?: { r: [number, number][]; g: [number, number][]; b: [number, number][] };
}

export interface CanvasControlAdapterContract {
  kind: 'controlnet' | 't2i_adapter' | 'control_lora' | 'z_image_control';
  model: string | null;
  weight: number;
  beginEndStepPct: [number, number];
  controlMode: 'balanced' | 'more_prompt' | 'more_control' | 'unbalanced' | null;
}

export interface CanvasMaskFillContract {
  style: 'solid' | 'grid' | 'crosshatch' | 'diagonal' | 'horizontal' | 'vertical';
  color: string;
}

export interface CanvasMaskContract {
  bitmap: CanvasImageRef | null;
  fill: CanvasMaskFillContract;
  /**
   * The layer-local origin of `bitmap`'s top-left pixel. Mask layers are
   * content-sized exactly like paint layers: the persisted mask bitmap covers
   * only the painted region, and this offset records where that region sits in
   * the layer's local space (it can be negative). Absent (or `{ x: 0, y: 0 }`)
   * for legacy documents whose mask bitmaps were document-sized at the origin —
   * they load identically. Mirrors {@link CanvasLayerSourceContract} `paint`.
   */
  offset?: { x: number; y: number };
}

export interface CanvasRasterLayerContractV2 extends CanvasLayerBaseContract {
  type: 'raster';
  source: CanvasLayerSourceContract;
  adjustments?: CanvasAdjustmentsContract;
  isTransparencyLocked?: boolean;
  filter?: { type: string; settings: Record<string, unknown> };
}

export interface CanvasControlLayerContract extends CanvasLayerBaseContract {
  type: 'control';
  source: CanvasLayerSourceContract;
  adapter: CanvasControlAdapterContract;
  withTransparencyEffect: boolean;
  filter?: { type: string; settings: Record<string, unknown> };
}

export interface CanvasRegionalGuidanceLayerContract extends CanvasLayerBaseContract {
  type: 'regional_guidance';
  mask: CanvasMaskContract;
  positivePrompt: string | null;
  negativePrompt: string | null;
  autoNegative: boolean;
  referenceImages: RegionalGuidanceReferenceImage[];
}

export interface CanvasInpaintMaskLayerContract extends CanvasLayerBaseContract {
  type: 'inpaint_mask';
  mask: CanvasMaskContract;
  noiseLevel?: number;
  denoiseLimit?: number;
}

export type CanvasLayerContract =
  | CanvasRasterLayerContractV2
  | CanvasControlLayerContract
  | CanvasRegionalGuidanceLayerContract
  | CanvasInpaintMaskLayerContract;

export interface CanvasDocumentContractV2 {
  version: 2;
  width: number;
  height: number;
  background: 'transparent' | { color: string };
  /** Index 0 is the top-most layer. */
  layers: CanvasLayerContract[];
  bbox: { x: number; y: number; width: number; height: number };
  selectedLayerId: string | null;
}

export interface CanvasSnapshotContract {
  id: string;
  name: string;
  createdAt: string;
  document: CanvasDocumentContractV2;
}

export interface CanvasStagingAreaContractV2 extends CanvasStagingAreaContract {
  autoSwitchMode: 'off' | 'latest' | 'progress';
}

export interface CanvasStateContractV2 {
  version: 2;
  document: CanvasDocumentContractV2;
  /**
   * Monotonic counter bumped whenever the document is swapped wholesale
   * (snapshot restore, `replaceCanvasDocument`) rather than incrementally
   * edited. The document mirror treats any change to this value as a full
   * document replacement (clearing engine pixel history), even when the new
   * document keeps the same dimensions and reuses layer ids — the case a
   * reference/dimension diff alone cannot distinguish from an ordinary edit.
   */
  documentRevision: number;
  snapshots: CanvasSnapshotContract[];
  stagingArea: CanvasStagingAreaContractV2;
}
