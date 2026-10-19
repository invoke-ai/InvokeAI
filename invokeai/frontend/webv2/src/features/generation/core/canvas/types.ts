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

/** Legacy "Scale Before Processing": how the bbox maps to the size the model denoises at. */
export type CanvasScaleMethod = 'none' | 'auto' | 'manual';

export interface CanvasScalingSettings {
  method: CanvasScaleMethod;
  /** The manual processing size; a null side falls back to the bbox's. */
  width: number | null;
  height: number | null;
}

export interface CanvasCompositingSettings {
  infillMethod: CanvasInfillMethod;
  infillTileSize: number;
  infillPatchmatchDownscaleSize: number;
  infillColorValue: { r: number; g: number; b: number; a: number };
  maskBlur: number;
  coherenceMode: CanvasCoherenceMode;
  coherenceMinDenoise: number;
  coherenceEdgeSize: number;
  /** Preserve transparency outside the generated mask instead of compositing over the source image. */
  outputOnlyMaskedRegions: boolean;
}

/** The legacy-parity generation modes a canvas invoke resolves to. */
export type CanvasGenerationMode = 'txt2img' | 'img2img' | 'inpaint' | 'outpaint';

/** The generation modes the pure canvas graph compiler supports (full matrix). */
export type CanvasCompileMode = CanvasGenerationMode;

/** Pure compiler inputs reference already-uploaded media. */
export interface CompileCanvasGraphInput {
  /** Prompts / steps / model-adjacent settings, reused verbatim from Generate. */
  settings: GenerateSettings;
  model: GenerateModelConfig;
  projectSettings: GenerationProjectSettings;
  /** Runtime-derived random-device metadata; defaults to the legacy CPU/CUDA value for direct callers. */
  randDevice?: string;
  /** The resolved canvas mode (from `canvasMode.ts`). */
  mode: CanvasCompileMode;
  /** Destination selects intermediate canvas output or durable gallery output. */
  destination: ResultDestination;
  /** The generation bounding box, in document space. Its size overrides settings dims. */
  bbox: Rect;
  /** The uploaded bbox composite (executor result). Required for image-referencing modes. */
  compositeImageName: string | null;
  /**
   * Uploaded grayscale mask: white keeps, dark inpaints. Required for inpaint; outpaint can derive its mask from
   * image alpha.
   */
  maskImageName?: string | null;
  /** An optional noise-level mask adds image noise before encoding. */
  noiseMaskImageName?: string | null;
  /** Denoising strength in (0, 1]. Consulted for `img2img` / `inpaint` / `outpaint`. */
  strength: number;
  /** Resolved infill / coherence / mask-blur knobs and output-compositing policy. */
  compositing: CanvasCompositingSettings;
  /** Processing-size policy; absent means the bbox snapped to the model grid. */
  scaling?: CanvasScalingSettings;
  /** Prevalidated per-layer composites, each uploaded separately. */
  controlLayers?: readonly ControlLayerGraphInput[];
  /** Resolved inputs for each supported region. */
  regionalGuidance?: readonly RegionalGuidanceInput[];
}

/** Carry the resolved mode so consumers do not recompute it. */
export interface CompiledCanvasGraph {
  backendGraph: BackendGraphContract;
  graph: GraphContract;
  negativePromptNodeId: string;
  positivePromptNodeId: string;
  seedNodeId: string;
  mode: CanvasCompileMode;
}
