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

export interface CanvasStagingAreaContract {
  sourceQueueItemId?: string;
  selectedLayerId?: string;
  pendingImageIds: string[];
  pendingImages: CanvasStagingCandidateContract[];
  selectedImageIndex: number;
  isVisible: boolean;
  areThumbnailsVisible: boolean;
}

// Top-first stack forests hold raster, control, vector, regional guidance and inpaint nodes. Leaves use transforms
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

/** Persist family and label for unresolved-font recovery; browser-only family aliases are never persisted. */
export interface CanvasTextFontRef {
  id: string;
  contentHash: string;
  family: string;
  label: string;
}

export type CanvasTextFontStyle = 'normal' | 'italic' | 'oblique';

/** Keyed by four-character OpenType axis tag. */
export type CanvasTextFontVariations = Readonly<Record<string, number>>;

/** By name, never by resolved URL. */
export interface CanvasImageRef {
  imageName: string;
  width: number;
  height: number;
  contentHash?: string;
}

export type ParametricShapeKind = 'rect' | 'ellipse' | 'triangle' | 'star';

export type CanvasLayerSourceContract =
  | {
      type: 'paint';
      bitmap: CanvasImageRef | null;
      /** Layer-local, possibly negative; absent or zero means a legacy document-sized bitmap at the origin. */
      offset?: { x: number; y: number };
    }
  | { type: 'image'; image: CanvasImageRef }
  | {
      type: 'text';
      content: string;
      fontFamily: string;
      fontSize: number;
      fontWeight: number;
      /** Omitted for built-in and legacy text. */
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
      /** Layer-local px across the `width`×`height` box, closed implicitly. */
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
      /** Layer-local extent; legacy absence defaults to document dimensions via `getSourceContentRect`. */
      width?: number;
      height?: number;
      /** Layer-local linear midpoint or radial center; absent means the extent center. */
      center?: { x: number; y: number };
      /** Absent fits the ramp to the extent. */
      span?: number;
    };

export type CanvasPixelSourceContract = Extract<CanvasLayerSourceContract, { type: 'image' | 'paint' }>;

/** Also the PSD `layerColor` vocabulary. */
export const CANVAS_COLOR_LABELS = ['red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray'] as const;
export type CanvasColorLabel = (typeof CANVAS_COLOR_LABELS)[number];

export interface CanvasLayerBaseContract {
  id: string;
  name: string;
  isEnabled: boolean;
  isLocked: boolean;
  opacity: number;
  blendMode: CanvasBlendMode;
  /** Display-only; absent means none. */
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
  /** Absent displays the type's name. */
  name?: string;
}

/** Applied in list order; disabled entries keep settings but skip renders. */
export type CanvasAdjustmentEntry =
  | (CanvasAdjustmentEntryBase & { type: 'brightness-contrast'; brightness: number; contrast: number })
  | (CanvasAdjustmentEntryBase & {
      type: 'exposure';
      /** −5 to +5, applied in linear light (2^stops). */
      stops: number;
    })
  | (CanvasAdjustmentEntryBase & {
      type: 'levels';
      /** 0–255 with `inBlack < inWhite`; `gamma` 1 is linear. */
      inBlack: number;
      inWhite: number;
      gamma: number;
      outBlack: number;
      outWhite: number;
      /** Absent means all three. */
      channel?: 'rgb' | 'r' | 'g' | 'b';
    })
  | (CanvasAdjustmentEntryBase & { type: 'curves'; curves: CanvasAdjustmentCurves })
  | (CanvasAdjustmentEntryBase & { type: 'hsl'; saturation: number })
  | (CanvasAdjustmentEntryBase & {
      type: 'hue';
      /** Degrees, −180 to 180. */
      rotation: number;
    })
  | (CanvasAdjustmentEntryBase & { type: 'invert' })
  | (CanvasAdjustmentEntryBase & {
      /** Alpha is untouched; fully transparent pixels stay transparent. */
      type: 'noise';
      /** Both share the standard deviation `amount` sets, so switching keeps perceived strength. */
      distribution: 'gaussian' | 'uniform';
      /** `chroma` varies OKLab a and b only, keeping lightness. */
      grain: 'monochrome' | 'color' | 'chroma';
      /** `midtones` scales strength by 4·Y′·(1 − Y′), Y′ the Rec. 709 luma entering the noise pass. */
      response: 'flat' | 'midtones';
      /** Reference-space pixels, 1–8; above 1 a Gaussian of σ = √(size² − 1) / 2 blurs the field, renormalized. */
      size: number;
      /** 0–1 standard deviation: `amount × 128` channel units, or `amount × 0.32` on OKLab a and b for chroma. */
      amount: number;
      /** uint32; noise is a pure function of seed, integer reference-space pixel coordinate and channel. */
      seed: number;
    })
  /** Backend Clarity/Texture/Contrast Dehaze parity; owner-wide neighborhoods. `amount` −200 to 200, clamped. */
  | (CanvasAdjustmentEntryBase & {
      /** `size`: feature size in pixels, 4–1000. */
      type: 'clarity';
      amount: number;
      size: number;
    })
  | (CanvasAdjustmentEntryBase & {
      /** `size`: pixels, 1–100. */
      type: 'texture';
      amount: number;
      size: number;
    })
  | (CanvasAdjustmentEntryBase & {
      /** Radius follows the alpha bounds' shorter side. */
      type: 'dehaze';
      amount: number;
    });

export type CanvasAdjustmentsContract = readonly CanvasAdjustmentEntry[];

export interface CanvasControlAdapterContract {
  kind:
    | 'controlnet'
    | 't2i_adapter'
    | 'control_lora'
    | 'z_image_control'
    | 'anima_lllite'
    | 'qwen_image_control'
    | 'flux2_control';
  model: string | null;
  weight: number;
  beginEndStepPct: [number, number];
  controlMode: 'balanced' | 'more_prompt' | 'more_control' | 'unbalanced' | null;
}

export interface CanvasMaskFillContract {
  style: 'solid' | 'grid' | 'crosshatch' | 'diagonal' | 'diagonal_left' | 'horizontal' | 'vertical';
  color: string;
}

export interface CanvasMaskContract {
  bitmap: CanvasImageRef | null;
  fill: CanvasMaskFillContract;
  /** Layer-local, possibly negative; absent or zero means a legacy document-sized mask. */
  offset?: { x: number; y: number };
}

/** Uses the raster layer's live alpha as inpaint coverage and stores only overlay fill; absent means never added. */
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

export interface CanvasBezierPointContract {
  anchor: { x: number; y: number };
  /** Layer-local coordinates, rather than offsets from the anchor. */
  inHandle: { x: number; y: number } | null;
  outHandle: { x: number; y: number } | null;
  type: 'corner' | 'smooth' | 'symmetric';
}

export interface CanvasBezierPathContract {
  id: string;
  name: string | null;
  isClosed: boolean;
  points: CanvasBezierPointContract[];
}

/** Editable guide geometry; no raster source and no contribution to image generation. */
export { DEFAULT_VECTOR_GUIDE_COLOR, DEFAULT_VECTOR_STROKE_COLOR, DEFAULT_VECTOR_STROKE_WIDTH } from './vectorDefaults';

export interface CanvasVectorLayerContract extends CanvasLayerBaseContract {
  type: 'vector';
  paths: CanvasBezierPathContract[];
  /** Display-only guide color; absent uses DEFAULT_VECTOR_GUIDE_COLOR. */
  guideColor?: string;
  fillColor?: string;
  strokeColor: string;
  strokeWidth: number;
  fillEnabled?: boolean;
  strokeEnabled?: boolean;
  traceTaperEnds?: boolean;
  traceTaper?: number;
  isHidden?: boolean;
}

/** What a control layer's image should control; a closed set, so an unknown value refuses the document. */
export const CONTROL_TYPES = ['pose', 'depth', 'edges', 'softedge', 'lineart', 'sketch', 'color', 'tile'] as const;
export type ControlType = (typeof CONTROL_TYPES)[number];

/** The type string stays open so newer detectors survive a load. */
export interface CanvasControlDetectionContract {
  type: string;
  settings: Record<string, unknown>;
}

/** Derivation invariants I1–I5 live in `document/controlDerivation.ts`. */
export interface CanvasControlLayerContract extends CanvasLayerBaseContract {
  type: 'control';
  source: CanvasLayerSourceContract;
  adapter: CanvasControlAdapterContract;
  withTransparencyEffect: boolean;
  /** Absent when the pixels are not a filter result. */
  filter?: CanvasControlDetectionContract;
  /** Display-only; absent means visible. Raster layers use `isEnabled` instead. */
  isHidden?: boolean;
  /** Absent means no type chosen; never written as null. */
  controlType?: ControlType;
  /** `null` means use the image as-is; absent means legacy (no choice). */
  detection?: CanvasControlDetectionContract | null;
  original?: CanvasPixelSourceContract;
  /** Maps `original`'s frame into `source`'s layer-local frame; absent means identity. */
  originalTransform?: CanvasLayerBaseContract['transform'];
}

/** A closed persisted set: a build that does not know a stored role refuses the canvas instead of rewriting it. */
export const CANVAS_REGIONAL_LAYOUT_KINDS = ['object', 'character', 'text'] as const;

/** `text` is the exact visible text; other kinds keep it so switching back restores it. */
export interface CanvasRegionalLayoutContract {
  kind: (typeof CANVAS_REGIONAL_LAYOUT_KINDS)[number];
  text?: string;
}

export interface CanvasRegionalGuidanceLayerContract extends CanvasLayerBaseContract {
  type: 'regional_guidance';
  mask: CanvasMaskContract;
  positivePrompt: string | null;
  negativePrompt: string | null;
  autoNegative: boolean;
  referenceImages: RegionalGuidanceReferenceImage[];
  /** Absent means an object region; mask-based models ignore it. */
  layout?: CanvasRegionalLayoutContract;
  /** Display-only, like {@link CanvasControlLayerContract.isHidden}. */
  isHidden?: boolean;
}

export interface CanvasMaskNoiseContract {
  level: number;
  isEnabled: boolean;
}

/** Disabled or absent uses the default limit. */
export interface CanvasMaskDenoiseContract {
  limit: number;
  isEnabled: boolean;
}

export interface CanvasInpaintMaskLayerContract extends CanvasLayerBaseContract {
  type: 'inpaint_mask';
  mask: CanvasMaskContract;
  /** Absent means never added. */
  noise?: CanvasMaskNoiseContract;
  denoise?: CanvasMaskDenoiseContract;
  /** Display-only, like {@link CanvasControlLayerContract.isHidden}. */

  isHidden?: boolean;
}

export type CanvasLayerContract =
  | CanvasVectorLayerContract
  | CanvasRasterLayerContractV2
  | CanvasControlLayerContract
  | CanvasRegionalGuidanceLayerContract
  | CanvasInpaintMaskLayerContract;

export type CanvasLayerStackKind = CanvasLayerContract['type'];

/** `isHidden` is valid only for overlay stacks; adjustments apply to the composited children. */
export interface CanvasGroupContract {
  id: string;
  type: 'group';
  name: string;
  isEnabled: boolean;
  isLocked: boolean;
  isHidden?: boolean;
  /** Raster-stack groups only; absent means 1. */
  opacity?: number;
  /** Raster-stack groups only; absent means 'normal'. */
  blendMode?: CanvasBlendMode;
  /** Display-only; absent means none. */
  colorLabel?: CanvasColorLabel;
  adjustments?: CanvasAdjustmentsContract;
  /** Index 0 is the top-most child. */
  children: CanvasNodeContract[];
}

export type CanvasNodeContract = CanvasLayerContract | CanvasGroupContract;

/** One top-first forest per stack; a node's stack is the forest it lives in. */
export type CanvasStackForests = Record<Exclude<CanvasLayerStackKind, 'vector'>, CanvasNodeContract[]> & {
  /** Absent until first used so projects without vectors retain their existing schema. */
  vector?: CanvasNodeContract[];
};

/** Bottom stack first; the one table every consumer reads. */
export const LAYER_STACK_ORDER: readonly CanvasLayerStackKind[] = [
  'raster',
  'control',
  'vector',
  'regional_guidance',
  'inpaint_mask',
];

export const LAYER_STACKS_TOP_FIRST: readonly CanvasLayerStackKind[] = [...LAYER_STACK_ORDER].reverse();

export const CANVAS_MAX_NODE_DEPTH = 10;
export const CANVAS_MAX_NODE_COUNT = 10_000;

export interface CanvasDocumentContractV3 {
  /**
   * Schema this document needs: 3 for plain content, 4 for custom-font typography, 5 for noise or spatial
   * (clarity, texture, dehaze) adjustments, 6 for control-layer derivation fields, 7 for vector geometry and
   * layer appearance, 8 for a Diagonal Left regenerate region on a raster layer. Each mutation
   * raises it to the newer of its content's need and the canvas state's version. A restored snapshot keeps its own,
   * possibly lower, version until the next mutation; the state's version and the project's compatibility floor stay
   * put meanwhile.
   */
  version: 3 | 4 | 5 | 6 | 7 | 8;
  width: number;
  height: number;
  background: 'transparent' | { color: string };
  stacks: CanvasStackForests;
  bbox: { x: number; y: number; width: number; height: number };
  /** Leaf-only tools refuse a group rather than guess a descendant. */
  selectedLayerId: string | null;
}

/** A missing kind restores like `clear-canvas`; a `discard-results` point lasts one document session. */
export type CanvasSnapshotKind = 'clear-canvas' | 'discard-results';

export interface CanvasSnapshotContract {
  id: string;
  name: string;
  createdAt: string;
  document: CanvasDocumentContractV3;
  kind?: CanvasSnapshotKind;
  stagedImages?: CanvasStagingCandidateContract[];
}

export interface CanvasStagingAreaContractV2 extends CanvasStagingAreaContract {
  /** Mirrors queue `AutoSwitchMode` without depending on it; `canvasMigration.ts` fails to compile if they differ. */
  autoSwitchMode: 'off' | 'latest' | 'progress';
}

export interface CanvasStateContractV3 {
  /**
   * Newest schema the state needs: the newest of its previous version, the live document's and every snapshot's
   * (3 plain, 4 custom-font typography, 5 noise or spatial adjustments, 6 control-layer derivation fields,
   * 7 vector geometry and layer appearance, 8 Diagonal Left raster regenerate regions). Sticky:
   * removing that content never lowers it.
   */
  version: 3 | 4 | 5 | 6 | 7 | 8;
  document: CanvasDocumentContractV3;
  /** Wholesale-replacement counter; the mirror clears pixel history on change, even with reused ids. */
  documentRevision: number;
  snapshots: CanvasSnapshotContract[];
  stagingArea: CanvasStagingAreaContractV2;
}
