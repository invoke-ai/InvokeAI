import type {
  CompiledGenerateGraph,
  ComponentModelConfig,
  GenerateLora,
  ImageWithDims,
  MainModelConfig,
  ModelIdentifierConfig,
  VaeModelConfig,
  VaePrecision,
} from '@features/generation/contracts';
import type { SeedMode } from '@platform/core/seed';

export type SpandrelModelConfig = ModelIdentifierConfig & {
  type: 'spandrel_image_to_image';
  /** The installed record's native scale; null or absent when it was installed before the probe recorded it. */
  scale?: number | null;
};
export type TileControlNetModelConfig = ModelIdentifierConfig & { type: 'controlnet' };

/**
 * Refine runs the upscaler and then a detail pass with a main model; Model only runs the upscaler alone. Values saved
 * before modes existed are Refine.
 */
export type UpscaleMode = 'refine' | 'model-only';

/** Project-persisted settings owned by the Upscale widget. */
export interface UpscaleWidgetValues {
  mode: UpscaleMode;
  inputImage: ImageWithDims | null;
  upscaleModel: SpandrelModelConfig | null;
  scale: number;
  creativity: number;
  structure: number;

  model: MainModelConfig | null;
  positivePrompt: string;
  positivePromptHeightPx: number;
  negativePrompt: string;
  negativePromptEnabled: boolean;
  negativePromptHeightPx: number;
  loras: GenerateLora[];
  steps: number;
  cfgScale: number;
  scheduler: string;
  batchCount: number;
  seed: number;
  seedMode: SeedMode;
  clipSkip: number;
  vae: VaeModelConfig | null;
  vaePrecision: VaePrecision;
  /** Set only where the main model's architecture asks for it (`upscaleComponentsFor`); null elsewhere. */
  t5EncoderModel: ComponentModelConfig | null;
  clipEmbedModel: ComponentModelConfig | null;
  qwen3EncoderModel: ComponentModelConfig | null;
  qwen35EncoderModel: ComponentModelConfig | null;
  qwen3VlEncoderModel: ComponentModelConfig | null;
  qwenVlEncoderModel: ComponentModelConfig | null;
  mistralEncoderModel: ComponentModelConfig | null;

  tileControlnetModel: TileControlNetModelConfig | null;
  tileSize: number;
  tileOverlap: number;
}

export interface CompiledUpscaleGraph extends CompiledGenerateGraph {
  outputNodeId: 'upscale_output';
}
