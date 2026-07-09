/**
 * Shared vocabulary for the canvas → generation-graph pipeline.
 *
 * This module is pure data: it names the generation mode a canvas invoke maps
 * to and the shape of a *composite plan* — a pixel-free description of what the
 * executor must composite and upload before a graph can be built. It has no
 * runtime dependency on the canvas engine (types are structural), so the
 * planner (`compositePlan.ts`) and mode detector (`canvasMode.ts`) stay pure,
 * node-testable, data-in/data-out functions.
 *
 * The plan is deliberately open-ended: Phase 4.2 only emits the single
 * `base-raster` composite, but later phases add `control-layer`,
 * `regional-mask`, and `inpaint-mask` composites to the same {@link CompositePlan}
 * structure — each a {@link CompositeEntry} with its own stable identity key.
 */

import type { GenerateModelConfig, GenerateSettings } from '@workbench/generation/types';
import type {
  BackendGraphContract,
  CanvasAdjustmentsContract,
  CanvasBlendMode,
  GraphContract,
  ProjectSettings,
  ResultDestination,
} from '@workbench/types';
import type { CanvasCompositingSettings } from '@workbench/widgets/canvas/invoke/canvasCompositing';

import type { ControlLayerGraphInput } from './addControlLayers';
import type { RegionalGuidanceInput } from './addRegionalGuidance';

/** The legacy-parity generation modes a canvas invoke resolves to. */
export type CanvasGenerationMode = 'txt2img' | 'img2img' | 'inpaint' | 'outpaint';

/**
 * Legacy default resolution for undefined mask attributes (`addInpaint.ts`):
 * a fresh mask with `denoiseLimit === undefined` denoises fully (1.0), and a
 * fresh mask with `noiseLevel === undefined` is EXCLUDED from the noise
 * composite entirely (no `img_noise`) — it must not be treated as noise 0.
 */
export const DEFAULT_MASK_DENOISE_LIMIT = 1;

/** An axis-aligned rectangle in document space (structurally the engine's `Rect`). */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A layer's 2D transform (translate / scale / rotate), mirrored from the contract. */
export interface CompositeLayerTransform {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
}

/**
 * A single layer's contribution to a composite entry — the frozen, pixel-
 * determining projection of a document layer. The executor draws these in
 * z-order; the planner derives them from the live document.
 */
export interface CompositeLayerRef {
  /** The document layer id (used to fetch its rasterized cache surface). */
  id: string;
  /**
   * A stable string identifying the layer's pixels: the resolved image name for
   * image/paint bitmaps (or a sentinel for an empty paint layer). Changing the
   * underlying asset changes this, and so changes the entry key.
   */
  sourceRef: string;
  /** Native (unscaled) pixel size of the layer's source, used for content-bounds math. */
  contentSize: { width: number; height: number };
  /**
   * Layer-local origin of the content (its cache surface's top-left) — non-zero
   * for content-sized paint layers whose persisted bitmap sits at an `offset`.
   * `{ x: 0, y: 0 }` for origin-anchored sources (image, legacy paint). The
   * executor draws the layer's cache at this local origin (then through the
   * transform), and content-bounds math projects the rect from here.
   */
  contentOffset: { x: number; y: number };
  transform: CompositeLayerTransform;
  opacity: number;
  blendMode: CanvasBlendMode;
  /**
   * Non-destructive raster adjustments (brightness/contrast/saturation/curves) to
   * bake into this layer's pixels before compositing, so the generated image uses
   * the ADJUSTED pixels the user sees. Present only for raster layers with a
   * non-identity adjustment; folded into the entry key so a changed adjustment
   * re-composites/uploads. Absent ⇒ raw pixels.
   */
  adjustments?: CanvasAdjustmentsContract;
}

/** The kind of composite an entry describes. */
export type CompositeEntryKind = 'base-raster' | 'inpaint-mask' | 'noise-mask' | 'control-layer' | 'regional-mask';

/**
 * A mask layer's contribution to a grayscale mask composite. The executor draws
 * each mask's alpha and colors it by `attributeValue` (white = keep, darker =
 * inpaint), darken-compositing multiple masks over a white background, exactly
 * like the legacy `getGrayscaleMaskCompositeImageDTO`.
 */
export interface CompositeMaskLayerRef {
  /** The document layer id (used to fetch its rasterized mask alpha surface). */
  id: string;
  /** Stable pixel identity (the mask bitmap name, or an empty sentinel). */
  sourceRef: string;
  contentSize: { width: number; height: number };
  contentOffset: { x: number; y: number };
  transform: CompositeLayerTransform;
  /**
   * The per-layer attribute value in [0, 1] driving the grayscale gray value
   * (`255 - round(255 * attributeValue)` for masked pixels). For the
   * `inpaint-mask` (denoise-limit) entry, an undefined `denoiseLimit` resolves
   * to {@link DEFAULT_MASK_DENOISE_LIMIT}; for the `noise-mask` entry it is the
   * layer's `noiseLevel` (only layers with a defined `noiseLevel` appear).
   */
  attributeValue: number;
}

/**
 * One composite the executor must produce: a set of layers composited over a
 * bbox into a single uploadable image. `key` is a content-identity hash of
 * everything that determines the entry's pixels (see {@link CompositePlan}).
 */
export interface CompositeEntry {
  kind: CompositeEntryKind;
  /**
   * Stable identity key: same document + bbox → same key, so the executor can
   * skip recompositing / re-uploading when nothing that affects these pixels
   * changed between invokes.
   */
  key: string;
  /** The bbox (document space) the entry is composited over and cropped to. */
  bbox: Rect;
  /** Contributing raster layers in document z-order (`base-raster` entries). */
  layers: CompositeLayerRef[];
  /** Contributing mask layers, for grayscale mask entries (`inpaint-mask` / `noise-mask`). */
  maskLayers?: CompositeMaskLayerRef[];
  /**
   * The single document layer id a `control-layer` entry composites (each
   * enabled control layer is composited SEPARATELY, never blended together).
   */
  layerId?: string;
}

/**
 * A pixel-free description of everything an invoke must composite/upload. Phase
 * 4.2 emits exactly one `base-raster` entry; later phases append more.
 */
export interface CompositePlan {
  /** The bbox (document space) the plan is scoped to. */
  bbox: Rect;
  entries: CompositeEntry[];
}

/** The generation modes the pure canvas graph compiler supports (full matrix). */
export type CanvasCompileMode = CanvasGenerationMode;

/**
 * Everything {@link import('./compileCanvasGraph').compileCanvasGraph} needs to
 * build a canvas generation graph. Pure data: the executor (Task 16) supplies
 * the already-uploaded `compositeImageName`; no pixels or engine state leak in.
 */
export interface CompileCanvasGraphInput {
  /** Prompts / steps / model-adjacent settings, reused verbatim from Generate. */
  settings: GenerateSettings;
  model: GenerateModelConfig;
  projectSettings: Pick<ProjectSettings, 'useCpuNoise'>;
  /** The resolved canvas mode (from `canvasMode.ts`). */
  mode: CanvasCompileMode;
  /**
   * The resolved result destination. `canvas` marks the output node intermediate
   * (staging); `gallery` makes it a durable image, matching `compileGenerateGraph`.
   */
  destination: ResultDestination;
  /** The generation bounding box, in document space. Its size overrides settings dims. */
  bbox: Rect;
  /** The uploaded bbox composite (executor result). Required for image-referencing modes. */
  compositeImageName: string | null;
  /**
   * The uploaded grayscale denoise-limit mask (white = keep, dark = inpaint).
   * Required for `inpaint` / `outpaint`.
   */
  maskImageName?: string | null;
  /**
   * The uploaded grayscale noise mask, present only when at least one enabled
   * mask layer defines a `noiseLevel`. Adds an `img_noise` node before encode.
   */
  noiseMaskImageName?: string | null;
  /** Denoising strength in (0, 1]. Consulted for `img2img` / `inpaint` / `outpaint`. */
  strength: number;
  /** Infill / coherence / mask-blur knobs. Defaults applied by the reader. */
  compositing?: CanvasCompositingSettings;
  /**
   * Valid, already-resolved control layers (each with its own uploaded composite
   * image name). The executor filters + composites these; the compiler only
   * grafts adapter nodes. Present in every mode (control works with all).
   */
  controlLayers?: readonly ControlLayerGraphInput[];
  /**
   * Valid, already-resolved regional-guidance regions (each with its own uploaded
   * mask image name + resolved reference-image models). The executor filters +
   * composites these; the compiler only grafts the mask-tensor + conditioning
   * nodes into the base graph's `pos_cond_collect` / `neg_cond_collect`. Applied
   * only for supported bases (sd-1 / sdxl / flux).
   */
  regionalGuidance?: readonly RegionalGuidanceInput[];
}

/**
 * Mirrors {@link import('../types').CompiledGenerateGraph} with the resolved
 * canvas mode attached so downstream (Task 18) can label / branch without
 * re-deriving it.
 */
export interface CompiledCanvasGraph {
  backendGraph: BackendGraphContract;
  graph: GraphContract;
  negativePromptNodeId: string;
  positivePromptNodeId: string;
  seedNodeId: string;
  mode: CanvasCompileMode;
}
