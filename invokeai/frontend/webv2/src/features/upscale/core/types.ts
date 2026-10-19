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

export type SpandrelModelConfig = ModelIdentifierConfig & { type: 'spandrel_image_to_image' };
export type TileControlNetModelConfig = ModelIdentifierConfig & { type: 'controlnet' };

/** Project-persisted settings owned by the Upscale widget. */
export interface UpscaleWidgetValues {
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

  tileControlnetModel: TileControlNetModelConfig | null;
  tileSize: number;
  tileOverlap: number;
}

export interface CompiledUpscaleGraph extends CompiledGenerateGraph {
  outputNodeId: 'upscale_output';
}
