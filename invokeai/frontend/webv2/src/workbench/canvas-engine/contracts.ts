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

/** The legacy-shaped staging area; v3 state extends it with `autoSwitchMode`. */
export interface CanvasStagingAreaContract {
  sourceQueueItemId?: string;
  selectedLayerId?: string;
  pendingImageIds: string[];
  pendingImages: CanvasStagingCandidateContract[];
  selectedImageIndex: number;
  isVisible: boolean;
  areThumbnailsVisible: boolean;
}

// Four top-first stack forests hold raster, control, regional guidance and inpaint nodes. Leaves use transforms
// and stable image names, never ephemeral URLs. Groups gate descendant enabled, locked and hidden state.

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

/**
 * Stable custom-font identity. Persist family and label for unresolved-font recovery; browser-only family aliases
 * are never persisted.
 */
export interface CanvasTextFontRef {
  id: string;
  contentHash: string;
  family: string;
  label: string;
}

/** CSS font style accepted by both CanvasRenderingContext2D and FontFace. */
export type CanvasTextFontStyle = 'normal' | 'italic' | 'oblique';

/** Explicit OpenType variation coordinates, keyed by four-character axis tag. */
export type CanvasTextFontVariations = Readonly<Record<string, number>>;

/** A reference to a persisted image asset by name, not by resolved URL. */
export interface CanvasImageRef {
  imageName: string;
  width: number;
  height: number;
  contentHash?: string;
}

/** The box-parametric shape kinds; `polygon` carries its own point list instead. */
export type ParametricShapeKind = 'rect' | 'ellipse' | 'triangle' | 'star';

export type CanvasLayerSourceContract =
  | {
      type: 'paint';
      bitmap: CanvasImageRef | null;
      /**
       * Bitmap origin in layer-local pixels, possibly negative. Paint bitmaps cover only content; absent or zero
       * preserves legacy document-sized bitmaps at the origin.
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
      /** Optional stable custom-font identity; omitted for built-in/legacy text. */
      fontRef?: CanvasTextFontRef;
      /** Defaults to `normal` for documents written before v4. */
      fontStyle?: CanvasTextFontStyle;
      /** Defaults to an empty coordinate set for documents written before v4. */
      fontVariations?: CanvasTextFontVariations;
      lineHeight: number;
      align: 'left' | 'center' | 'right';
      color: string;
    }
  | {
      type: 'shape';
      kind: ParametricShapeKind | 'polygon';
      /** Polygon vertices in layer-local px across the `width`×`height` box, closed implicitly. */
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
       * Layer-local gradient extent, initialized from the bbox and preserved across angle edits. Legacy absence
       * defaults to document dimensions via `getSourceContentRect`.
       */
      width?: number;
      height?: number;
      /**
       * Where the gradient sits in the extent (layer-local px): the midpoint of
       * a linear ramp, or the radial center. Absent = the extent center.
       */
      center?: { x: number; y: number };
      /** Linear stop distance or radial radius; absence fits the ramp to the extent. */
      span?: number;
    };

/** Photoshop's layer-color palette; also the PSD `layerColor` vocabulary. */
export const CANVAS_COLOR_LABELS = ['red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray'] as const;
export type CanvasColorLabel = (typeof CANVAS_COLOR_LABELS)[number];

export interface CanvasLayerBaseContract {
  id: string;
  name: string;
  isEnabled: boolean;
  isLocked: boolean;
  opacity: number;
  blendMode: CanvasBlendMode;
  /** Organizational color label; display-only, absent means none. */
  colorLabel?: CanvasColorLabel;
  transform: { x: number; y: number; scaleX: number; scaleY: number; rotation: number };
}

export interface CanvasAdjustmentCurves {
  r?: [number, number][];
  g?: [number, number][];
  b?: [number, number][];
}

interface CanvasAdjustmentEntryBase {
  id: string;
  isEnabled: boolean;
  /** A user-given name; absent entries display their type's name. */
  name?: string;
}

/**
 * Non-destructive raster adjustment, applied in list order. Disabling preserves settings while excluding it from
 * renders.
 */
export type CanvasAdjustmentEntry =
  | (CanvasAdjustmentEntryBase & { type: 'brightness-contrast'; brightness: number; contrast: number })
  | (CanvasAdjustmentEntryBase & {
      type: 'exposure';
      /** Photographic stops, −5 to +5, applied in linear light (2^stops). */
      stops: number;
    })
  | (CanvasAdjustmentEntryBase & {
      type: 'levels';
      /** Input remap: 0–255 with `inBlack < inWhite`; `gamma` is the midtone exponent base (1 = linear). */
      inBlack: number;
      inWhite: number;
      gamma: number;
      outBlack: number;
      outWhite: number;
      /** Which channels the remap drives; absent ⇒ all three. */
      channel?: 'rgb' | 'r' | 'g' | 'b';
    })
  | (CanvasAdjustmentEntryBase & { type: 'curves'; curves: CanvasAdjustmentCurves })
  | (CanvasAdjustmentEntryBase & { type: 'hsl'; saturation: number })
  | (CanvasAdjustmentEntryBase & {
      type: 'hue';
      /** Rotation around the color wheel in degrees, -180 to 180. */
      rotation: number;
    })
  | (CanvasAdjustmentEntryBase & { type: 'invert' });

/** A raster layer's ordered adjustment stack, applied top to bottom. */
export type CanvasAdjustmentsContract = readonly CanvasAdjustmentEntry[];

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
   * Content-sized mask bitmap origin in layer-local pixels, possibly negative. Absent or zero preserves legacy
   * document-sized masks; matches {@link CanvasLayerSourceContract} paint offsets.
   */
  offset?: { x: number; y: number };
}

/**
 * Singleton regenerate region using the raster layer's live alpha as inpaint coverage. It follows strokes, erasure
 * and transforms, contributes while the layer does, and stores only overlay fill. Absence means never added.
 */
export interface CanvasLayerRegionContract {
  isEnabled: boolean;
  name?: string;
  fill: CanvasMaskFillContract;
}

export interface CanvasRasterLayerContractV2 extends CanvasLayerBaseContract {
  type: 'raster';
  source: CanvasLayerSourceContract;
  adjustments?: CanvasAdjustmentsContract;
  inpaint?: CanvasLayerRegionContract;
  isTransparencyLocked?: boolean;
  filter?: { type: string; settings: Record<string, unknown> };
}

export interface CanvasControlLayerContract extends CanvasLayerBaseContract {
  type: 'control';
  source: CanvasLayerSourceContract;
  adapter: CanvasControlAdapterContract;
  withTransparencyEffect: boolean;
  filter?: { type: string; settings: Record<string, unknown> };
  /**
   * Display-only visibility for overlay layers; generation is unchanged. Absence means visible. Raster layers
   * instead use `isEnabled` for both visibility and participation.
   */
  isHidden?: boolean;
}

export interface CanvasRegionalGuidanceLayerContract extends CanvasLayerBaseContract {
  type: 'regional_guidance';
  mask: CanvasMaskContract;
  positivePrompt: string | null;
  negativePrompt: string | null;
  autoNegative: boolean;
  referenceImages: RegionalGuidanceReferenceImage[];
  /**
   * Display-only visibility for overlay layers; generation is unchanged. Absence means visible. Raster layers
   * instead use `isEnabled` for both visibility and participation.
   */
  isHidden?: boolean;
}

/** A mask's noise modifier; disabling keeps the tuned level out of generation without losing it. */
export interface CanvasMaskNoiseContract {
  level: number;
  isEnabled: boolean;
}

/** A mask's denoise-limit modifier; disabled or absent, generation uses the default limit. */
export interface CanvasMaskDenoiseContract {
  limit: number;
  isEnabled: boolean;
}

export interface CanvasInpaintMaskLayerContract extends CanvasLayerBaseContract {
  type: 'inpaint_mask';
  mask: CanvasMaskContract;
  /** Absent ⇒ the modifier was never added; the Layers tree projects it as a child row. */
  noise?: CanvasMaskNoiseContract;
  denoise?: CanvasMaskDenoiseContract;
  /**
   * Display-only visibility for overlay layers; generation is unchanged. Absence means visible. Raster layers
   * instead use `isEnabled` for both visibility and participation.
   */
  isHidden?: boolean;
}

export type CanvasLayerContract =
  | CanvasRasterLayerContractV2
  | CanvasControlLayerContract
  | CanvasRegionalGuidanceLayerContract
  | CanvasInpaintMaskLayerContract;

export type CanvasLayerStackKind = CanvasLayerContract['type'];

/**
 * Groups contain only their own stack's nodes. `isHidden` is display-only and valid only for overlay stacks.
 * Raster-only adjustments apply to composited children before the parent; identity adjustments pass through.
 */
export interface CanvasGroupContract {
  id: string;
  type: 'group';
  name: string;
  isEnabled: boolean;
  isLocked: boolean;
  isHidden?: boolean;
  /** Raster-stack groups only; absent means 1. Applies to the group's isolated composite. */
  opacity?: number;
  /** Raster-stack groups only; absent means 'normal'. Applies to the group's isolated composite. */
  blendMode?: CanvasBlendMode;
  /** Organizational color label; display-only, absent means none. */
  colorLabel?: CanvasColorLabel;
  adjustments?: CanvasAdjustmentsContract;
  /** Index 0 is the top-most child. */
  children: CanvasNodeContract[];
}

export type CanvasNodeContract = CanvasLayerContract | CanvasGroupContract;

/** One top-first forest per stack; a node's stack is the forest it lives in. */
export type CanvasStackForests = Record<CanvasLayerStackKind, CanvasNodeContract[]>;

/** The composition order, bottom stack first; the one table every consumer reads. */
export const LAYER_STACK_ORDER: readonly CanvasLayerStackKind[] = [
  'raster',
  'control',
  'regional_guidance',
  'inpaint_mask',
];

/** The same stacks as the panel lists them, top first. */
export const LAYER_STACKS_TOP_FIRST: readonly CanvasLayerStackKind[] = [...LAYER_STACK_ORDER].reverse();

export const CANVAS_MAX_NODE_DEPTH = 10;
export const CANVAS_MAX_NODE_COUNT = 10_000;

export interface CanvasDocumentContractV3 {
  /** v3 is read for compatibility; v4 is the current writable schema. */
  version: 3 | 4;
  width: number;
  height: number;
  background: 'transparent' | { color: string };
  stacks: CanvasStackForests;
  bbox: { x: number; y: number; width: number; height: number };
  /** A leaf or a group; leaf-only tools refuse a group rather than guessing a descendant. */
  selectedLayerId: string | null;
}

/** Current writable document contract. v3 remains accepted at the load boundary. */
export type CanvasDocumentContractV4 = Omit<CanvasDocumentContractV3, 'version'> & { version: 4 };

export interface CanvasSnapshotContract {
  id: string;
  name: string;
  createdAt: string;
  document: CanvasDocumentContractV3;
}

export interface CanvasStagingAreaContractV2 extends CanvasStagingAreaContract {
  autoSwitchMode: 'off' | 'latest' | 'progress';
}

export interface CanvasStateContractV3 {
  /** v3 is read for compatibility; v4 is the current writable schema. */
  version: 3 | 4;
  document: CanvasDocumentContractV3;
  /**
   * Monotonic wholesale-replacement counter. The mirror clears pixel history whenever it changes, including
   * replacements reusing dimensions and layer ids that ordinary diffs cannot detect.
   */
  documentRevision: number;
  snapshots: CanvasSnapshotContract[];
  stagingArea: CanvasStagingAreaContractV2;
}

/** Current writable state contract. v3 remains accepted at the load boundary. */
export type CanvasStateContractV4 = Omit<CanvasStateContractV3, 'version'> & { version: 4 };
