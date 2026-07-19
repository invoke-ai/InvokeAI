import type {
  BackendGraphContract,
  GenerationProjectSettings,
  GraphContract,
  ResultDestination,
} from '@features/generation/core/contracts';
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import type { ControlLayerGraphInput } from './addControlLayers';
import type { RegionalGuidanceInput } from './addRegionalGuidance';

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type CanvasInfillMethod = 'patchmatch' | 'lama' | 'cv2' | 'color' | 'tile';
export type CanvasCoherenceMode = 'Gaussian Blur' | 'Box Blur' | 'Staged';

export interface CanvasCompositingSettings {
  infillMethod: CanvasInfillMethod;
  infillTileSize: number;
  infillPatchmatchDownscaleSize: number;
  infillColorValue: { r: number; g: number; b: number; a: number };
  maskBlur: number;
  coherenceMode: CanvasCoherenceMode;
  coherenceMinDenoise: number;
  coherenceEdgeSize: number;
}

export const DEFAULT_CANVAS_COMPOSITING: CanvasCompositingSettings = {
  coherenceEdgeSize: 16,
  coherenceMinDenoise: 0,
  coherenceMode: 'Gaussian Blur',
  infillColorValue: { a: 1, b: 0, g: 0, r: 0 },
  infillMethod: 'lama',
  infillPatchmatchDownscaleSize: 1,
  infillTileSize: 32,
  maskBlur: 16,
};

/** The legacy-parity generation modes a canvas invoke resolves to. */
export type CanvasGenerationMode = 'txt2img' | 'img2img' | 'inpaint' | 'outpaint';

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
  projectSettings: GenerationProjectSettings;
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
