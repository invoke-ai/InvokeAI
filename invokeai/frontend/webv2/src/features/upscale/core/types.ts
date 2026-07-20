import type {
  CompiledGenerateGraph,
  GenerateLora,
  ImageWithDims,
  MainModelConfig,
  ModelIdentifierConfig,
  VaeModelConfig,
  VaePrecision,
} from '@features/generation/contracts';

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
  shouldRandomizeSeed: boolean;
  clipSkip: number;
  vae: VaeModelConfig | null;
  vaePrecision: VaePrecision;

  tileControlnetModel: TileControlNetModelConfig | null;
  tileSize: number;
  tileOverlap: number;
}

export interface CompiledUpscaleGraph extends CompiledGenerateGraph {
  outputNodeId: 'upscale_output';
}
