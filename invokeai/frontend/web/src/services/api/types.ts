import type { Dimensions } from 'features/controlLayers/store/types';
import type { components, paths } from 'services/api/schema';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import type { JsonObject, SetRequired } from 'type-fest';
import z from 'zod';

export type S = components['schemas'];

export type ListImagesArgs = NonNullable<paths['/api/v1/images/']['get']['parameters']['query']>;
export type ListImagesResponse = paths['/api/v1/images/']['get']['responses']['200']['content']['application/json'];

export type GetImageNamesResult =
  paths['/api/v1/images/names']['get']['responses']['200']['content']['application/json'];
export type GetImageNamesArgs = NonNullable<paths['/api/v1/images/names']['get']['parameters']['query']>;

export type ListBoardsArgs = NonNullable<paths['/api/v1/boards/']['get']['parameters']['query']>;

export type CreateBoardArg = paths['/api/v1/boards/']['post']['parameters']['query'];

export type UpdateBoardArg = paths['/api/v1/boards/{board_id}']['patch']['parameters']['path'] & {
  changes: paths['/api/v1/boards/{board_id}']['patch']['requestBody']['content']['application/json'];
};

export type GraphAndWorkflowResponse =
  paths['/api/v1/images/i/{image_name}/workflow']['get']['responses']['200']['content']['application/json'];

export type EnqueueBatchArg =
  paths['/api/v1/queue/{queue_id}/enqueue_batch']['post']['requestBody']['content']['application/json'];

export type GetQueueItemIdsResult =
  paths['/api/v1/queue/{queue_id}/item_ids']['get']['responses']['200']['content']['application/json'];
export type GetQueueItemIdsArgs = NonNullable<paths['/api/v1/queue/{queue_id}/item_ids']['get']['parameters']['query']>;

export type GetQueueItemSummariesByItemIdsResult =
  paths['/api/v1/queue/{queue_id}/item_summaries_by_ids']['post']['responses']['200']['content']['application/json'];
export type GetQueueItemSummariesByItemIdsArgs =
  paths['/api/v1/queue/{queue_id}/item_summaries_by_ids']['post']['requestBody']['content']['application/json'];

export type InputFieldJSONSchemaExtra = S['InputFieldJSONSchemaExtra'];
export type OutputFieldJSONSchemaExtra = S['OutputFieldJSONSchemaExtra'];
export type InvocationJSONSchemaExtra = S['UIConfigBase'];

// App Info
export type AppVersion = S['AppVersion'];
export type ExternalProviderStatus = {
  provider_id: string;
  configured: boolean;
  message?: string | null;
};
export type ExternalProviderConfig = {
  provider_id: string;
  api_key_configured: boolean;
  base_url?: string | null;
  message?: string | null;
};
export type ExternalProviderConfigUpdate = {
  api_key?: string;
  base_url?: string | null;
};
export type UpdateModelBody = paths['/api/v2/models/i/{key}']['patch']['requestBody']['content']['application/json'];

const zResourceOrigin = z.enum(['internal', 'external']);
type ResourceOrigin = z.infer<typeof zResourceOrigin>;
assert<Equals<ResourceOrigin, S['ResourceOrigin']>>();
const zImageCategory = z.enum(['general', 'mask', 'control', 'user', 'other']);
export type ImageCategory = z.infer<typeof zImageCategory>;
assert<Equals<ImageCategory, S['ImageCategory']>>();

// Images
const _zImageDTO = z.object({
  image_name: z.string(),
  image_url: z.string(),
  thumbnail_url: z.string(),
  image_origin: zResourceOrigin,
  image_category: zImageCategory,
  width: z.number().int().gt(0),
  height: z.number().int().gt(0),
  created_at: z.string(),
  updated_at: z.string(),
  deleted_at: z.string().nullish(),
  is_intermediate: z.boolean(),
  session_id: z.string().nullish(),
  node_id: z.string().nullish(),
  starred: z.boolean(),
  has_workflow: z.boolean(),
  board_id: z.string().nullish(),
  image_subfolder: z.string().optional(),
});
export type ImageDTO = z.infer<typeof _zImageDTO>;
assert<Equals<ImageDTO, S['ImageDTO']>>();

export type BoardDTO = S['BoardDTO'];
export type OffsetPaginatedResults_ImageDTO_ = S['OffsetPaginatedResults_ImageDTO_'];

// Model Configs
type InternalAnyModelConfig = S['AnyModelConfig'];
export type MainModelConfig = Extract<InternalAnyModelConfig, { type: 'main' }>;
type FLUXModelConfig = Extract<InternalAnyModelConfig, { type: 'main'; base: 'flux' }>;
type FLUX2ModelConfig = Extract<InternalAnyModelConfig, { type: 'main'; base: 'flux2' }>;
export type AnyFLUXModelConfig = FLUXModelConfig | FLUX2ModelConfig;
export type ControlLoRAModelConfig = Extract<InternalAnyModelConfig, { type: 'control_lora' }>;
export type LoRAModelConfig = Extract<InternalAnyModelConfig, { type: 'lora' }>;
type WanLoRAModelConfig = Extract<InternalAnyModelConfig, { type: 'lora'; base: 'wan' }>;
export type VAEModelConfig = Extract<InternalAnyModelConfig, { type: 'vae' }>;
export type ControlNetModelConfig = Extract<InternalAnyModelConfig, { type: 'controlnet' }>;
type AnimaControlNetModelConfig = Extract<InternalAnyModelConfig, { type: 'controlnet'; base: 'anima' }>;
export type IPAdapterModelConfig = Extract<InternalAnyModelConfig, { type: 'ip_adapter' }>;
export type T2IAdapterModelConfig = Extract<InternalAnyModelConfig, { type: 't2i_adapter' }>;
export type CLIPLEmbedModelConfig = Extract<InternalAnyModelConfig, { type: 'clip_embed'; variant: 'large' }>;
export type CLIPGEmbedModelConfig = Extract<InternalAnyModelConfig, { type: 'clip_embed'; variant: 'gigantic' }>;
export type CLIPEmbedModelConfig = Extract<InternalAnyModelConfig, { type: 'clip_embed' }>;
export type LlavaOnevisionModelConfig = Extract<InternalAnyModelConfig, { type: 'llava_onevision' }>;
export type TextLLMModelConfig = Extract<InternalAnyModelConfig, { type: 'text_llm' }>;
export type T5EncoderModelConfig = Extract<InternalAnyModelConfig, { type: 't5_encoder' }>;
export type T5EncoderBnbQuantizedLlmInt8bModelConfig = Extract<
  InternalAnyModelConfig,
  { type: 't5_encoder'; format: 'bnb_quantized_int8b' }
>;
export type Qwen3EncoderModelConfig = Extract<InternalAnyModelConfig, { type: 'qwen3_encoder' }>;
export type MistralEncoderModelConfig = Extract<InternalAnyModelConfig, { type: 'mistral_encoder' }>;
export type QwenVLEncoderModelConfig = Extract<InternalAnyModelConfig, { type: 'qwen_vl_encoder' }>;
export type Qwen3VLEncoderModelConfig = Extract<InternalAnyModelConfig, { type: 'qwen3_vl_encoder' }>;
export type WanT5EncoderModelConfig = Extract<InternalAnyModelConfig, { type: 'wan_t5_encoder' }>;
type Gemma2EncoderModelConfig = Extract<InternalAnyModelConfig, { type: 'gemma2_encoder' }>;
type PiDDecoderModelConfig = Extract<InternalAnyModelConfig, { type: 'pid_decoder' }>;
export type SpandrelImageToImageModelConfig = Extract<InternalAnyModelConfig, { type: 'spandrel_image_to_image' }>;
export type CheckpointModelConfig = Extract<InternalAnyModelConfig, { type: 'main'; format: 'checkpoint' }>;
export type CLIPVisionModelConfig = Extract<InternalAnyModelConfig, { type: 'clip_vision' }>;
export type SigLIPModelConfig = Extract<InternalAnyModelConfig, { type: 'siglip' }>;
export type FLUXReduxModelConfig = Extract<InternalAnyModelConfig, { type: 'flux_redux' }>;
type ApiModelConfig = Extract<InternalAnyModelConfig, { format: 'api' }>;
type UnknownModelConfig = Extract<InternalAnyModelConfig, { type: 'unknown' }>;
export type FLUXKontextModelConfig = MainModelConfig;
export type ChatGPT4oModelConfig = ApiModelConfig;
export type Gemini2_5ModelConfig = ApiModelConfig;
type SubmodelDefinition = S['SubmodelDefinition'];

export type ExternalImageSize = {
  width: number;
  height: number;
};

type ExternalResolutionPreset = {
  label: string;
  aspect_ratio: string;
  image_size: string;
  width: number;
  height: number;
};

export type ExternalModelCapabilities = {
  modes: ('txt2img' | 'img2img' | 'inpaint')[];
  supports_reference_images?: boolean;
  supports_negative_prompt?: boolean;
  supports_seed?: boolean;
  supports_guidance?: boolean;
  supports_steps?: boolean;
  max_images_per_request?: number | null;
  max_image_size?: ExternalImageSize | null;
  allowed_aspect_ratios?: string[] | null;
  aspect_ratio_sizes?: Record<string, ExternalImageSize> | null;
  resolution_presets?: ExternalResolutionPreset[] | null;
  max_reference_images?: number | null;
  mask_format?: 'alpha' | 'binary' | 'none';
  input_image_required_for?: ('txt2img' | 'img2img' | 'inpaint')[] | null;
};

export type ExternalApiModelDefaultSettings = {
  width?: number | null;
  height?: number | null;
  steps?: number | null;
  guidance?: number | null;
  num_images?: number | null;
};

export type ExternalPanelControlName =
  | 'negative_prompt'
  | 'reference_images'
  | 'dimensions'
  | 'seed'
  | 'steps'
  | 'guidance';

export type ExternalModelPanelControl = {
  name: ExternalPanelControlName;
  slider_min?: number | null;
  slider_max?: number | null;
  number_input_min?: number | null;
  number_input_max?: number | null;
  fine_step?: number | null;
  coarse_step?: number | null;
  marks?: number[] | null;
};

export type ExternalModelPanelSchema = {
  prompts: ExternalModelPanelControl[];
  image: ExternalModelPanelControl[];
  generation: ExternalModelPanelControl[];
};

export type ExternalApiModelConfig = {
  key: string;
  hash: string;
  path: string;
  file_size: number;
  name: string;
  description: string | null;
  source: string;
  source_type: string;
  source_api_response: JsonObject | null;
  cover_image: string | null;
  base: 'external';
  type: 'external_image_generator';
  format: 'external_api';
  provider_id: string;
  provider_model_id: string;
  capabilities: ExternalModelCapabilities;
  default_settings?: ExternalApiModelDefaultSettings | null;
  panel_schema?: ExternalModelPanelSchema | null;
  tags?: string[] | null;
  is_default?: boolean;
};
export type AnyModelConfig = InternalAnyModelConfig;
export type AnyModelConfigWithExternal = AnyModelConfig | ExternalApiModelConfig;
export type MainOrExternalModelConfig = MainModelConfig | ExternalApiModelConfig;

/**
 * Checks if a list of submodels contains any that match a given variant or type
 * @param submodels The list of submodels to check
 * @param checkStr The string to check against for variant or type
 * @returns A boolean
 */
const checkSubmodel = (submodels: Record<string, SubmodelDefinition>, checkStr: string): boolean => {
  for (const submodel in submodels) {
    if (
      submodel &&
      submodels[submodel] &&
      (submodels[submodel].model_type === checkStr || submodels[submodel].variant === checkStr)
    ) {
      return true;
    }
  }
  return false;
};

/**
 * Checks if a main model config has submodels that match a given variant or type
 * @param identifiers A list of strings to check against for variant or type in submodels
 * @param config The model config
 * @returns A boolean
 */
const checkSubmodels = (identifiers: string[], config: AnyModelConfig): boolean => {
  return identifiers.every(
    (identifier) =>
      config.type === 'main' &&
      'submodels' in config &&
      config.submodels &&
      (identifier in config.submodels || checkSubmodel(config.submodels, identifier))
  );
};

export const isLoRAModelConfig = (config: AnyModelConfig): config is LoRAModelConfig => {
  return config.type === 'lora';
};

export const isControlLoRAModelConfig = (config: AnyModelConfig): config is ControlLoRAModelConfig => {
  return config.type === 'control_lora';
};

export const isVAEModelConfigOrSubmodel = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is VAEModelConfig => {
  return config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config));
};

export const isVAEModelConfig = (config: AnyModelConfig): config is VAEModelConfig => {
  return config.type === 'vae';
};

export const isNonFluxVAEModelConfig = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is VAEModelConfig => {
  return (
    (config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config))) &&
    config.base !== 'flux' &&
    config.base !== 'flux2'
  );
};

export const isFluxVAEModelConfig = (config: AnyModelConfig, excludeSubmodels?: boolean): config is VAEModelConfig => {
  return (
    (config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config))) &&
    (config.base === 'flux' || config.base === 'flux2')
  );
};

export const isFlux1VAEModelConfig = (config: AnyModelConfig, excludeSubmodels?: boolean): config is VAEModelConfig => {
  return (
    (config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config))) &&
    config.base === 'flux'
  );
};

export const isFlux2VAEModelConfig = (config: AnyModelConfig, excludeSubmodels?: boolean): config is VAEModelConfig => {
  return (
    (config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config))) &&
    config.base === 'flux2'
  );
};

export const isAnimaVAEModelConfig = (config: AnyModelConfig, excludeSubmodels?: boolean): config is VAEModelConfig => {
  return (
    (config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config))) &&
    config.base === 'anima'
  );
};

export const isQwenImageVAEModelConfig = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is VAEModelConfig => {
  return (
    (config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config))) &&
    config.base === 'qwen-image'
  );
};

export const isWanVAEModelConfig = (config: AnyModelConfig, excludeSubmodels?: boolean): config is VAEModelConfig => {
  return (
    (config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config))) &&
    config.base === 'wan'
  );
};

export const isControlNetModelConfig = (config: AnyModelConfig): config is ControlNetModelConfig => {
  return config.type === 'controlnet';
};

export const isAnimaControlNetModelConfig = (config: AnyModelConfig): config is AnimaControlNetModelConfig => {
  return config.type === 'controlnet' && config.base === 'anima';
};

export const isAnimaInpaintControlNetModelConfig = (config: AnyModelConfig): config is AnimaControlNetModelConfig => {
  // 4-channel (RGB + mask) variants are inpaint adapters. Models with a null cond_in_channels were installed before
  // the field was recorded - only the inpaint variant predates it, so treat them as inpaint adapters too. (A 3ch
  // model installed from a pre-release dev build also reads as null and must be reinstalled to register as a
  // control layer model.)
  return isAnimaControlNetModelConfig(config) && config.cond_in_channels !== 3;
};

export const isControlLayerModelConfig = (
  config: AnyModelConfig
): config is ControlNetModelConfig | T2IAdapterModelConfig | ControlLoRAModelConfig => {
  if (isAnimaControlNetModelConfig(config)) {
    // Only the 3-channel (general control) Anima ControlNet-LLLite variants are usable as control layers; 4-channel
    // and legacy (null cond_in_channels) variants are inpaint adapters.
    return config.cond_in_channels === 3;
  }
  return config.type === 'controlnet' || config.type === 't2i_adapter' || config.type === 'control_lora';
};

export const isIPAdapterModelConfig = (config: AnyModelConfig): config is IPAdapterModelConfig => {
  return config.type === 'ip_adapter';
};

export const isCLIPVisionModelConfig = (config: AnyModelConfig): config is CLIPVisionModelConfig => {
  return config.type === 'clip_vision';
};

export const isLLaVAModelConfig = (config: AnyModelConfig): config is LlavaOnevisionModelConfig => {
  return config.type === 'llava_onevision';
};

export const isTextLLMModelConfig = (config: AnyModelConfig): config is TextLLMModelConfig => {
  return config.type === 'text_llm';
};

export const isT2IAdapterModelConfig = (config: AnyModelConfig): config is T2IAdapterModelConfig => {
  return config.type === 't2i_adapter';
};

export const isT5EncoderModelConfigOrSubmodel = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is T5EncoderModelConfig | T5EncoderBnbQuantizedLlmInt8bModelConfig => {
  return (
    config.type === 't5_encoder' ||
    (!excludeSubmodels && config.type === 'main' && checkSubmodels(['t5_encoder'], config))
  );
};

export const isT5EncoderModelConfig = (
  config: AnyModelConfig
): config is T5EncoderModelConfig | T5EncoderBnbQuantizedLlmInt8bModelConfig => {
  return config.type === 't5_encoder';
};

export const isQwen3EncoderModelConfig = (config: AnyModelConfig): config is Qwen3EncoderModelConfig => {
  return config.type === 'qwen3_encoder' && config.variant !== 'qwen3_06b';
};

export const isAnimaQwen3EncoderModelConfig = (config: AnyModelConfig): config is Qwen3EncoderModelConfig => {
  return config.type === 'qwen3_encoder' && config.variant === 'qwen3_06b';
};

export const isMistralEncoderModelConfig = (config: AnyModelConfig): config is MistralEncoderModelConfig => {
  return config.type === 'mistral_encoder';
};

export const isQwenVLEncoderModelConfig = (config: AnyModelConfig): config is QwenVLEncoderModelConfig => {
  return config.type === 'qwen_vl_encoder';
};

export const isQwen3VLEncoderModelConfig = (config: AnyModelConfig): config is Qwen3VLEncoderModelConfig => {
  return config.type === 'qwen3_vl_encoder';
};

export const isWanT5EncoderModelConfig = (config: AnyModelConfig): config is WanT5EncoderModelConfig => {
  return config.type === 'wan_t5_encoder';
};

export const isGemma2EncoderModelConfig = (config: AnyModelConfig): config is Gemma2EncoderModelConfig => {
  return config.type === 'gemma2_encoder';
};

export const isPiDDecoderModelConfig = (config: AnyModelConfig): config is PiDDecoderModelConfig => {
  return config.type === 'pid_decoder';
};

export const isCLIPEmbedModelConfigOrSubmodel = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is CLIPEmbedModelConfig => {
  return (
    config.type === 'clip_embed' ||
    (!excludeSubmodels && config.type === 'main' && checkSubmodels(['clip_embed'], config))
  );
};

export const isCLIPEmbedModelConfig = (config: AnyModelConfig): config is CLIPEmbedModelConfig => {
  return config.type === 'clip_embed';
};

export const isCLIPLEmbedModelConfigOrSubmodel = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is CLIPLEmbedModelConfig => {
  return (
    (config.type === 'clip_embed' && config.variant === 'large') ||
    (!excludeSubmodels && config.type === 'main' && checkSubmodels(['clip_embed', 'large'], config))
  );
};

export const isCLIPGEmbedModelConfigOrSubmodel = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is CLIPGEmbedModelConfig => {
  return (
    (config.type === 'clip_embed' && config.variant === 'gigantic') ||
    (!excludeSubmodels && config.type === 'main' && checkSubmodels(['clip_embed', 'gigantic'], config))
  );
};

export const isSpandrelImageToImageModelConfig = (
  config: AnyModelConfig
): config is SpandrelImageToImageModelConfig => {
  return config.type === 'spandrel_image_to_image';
};

export const isSigLipModelConfig = (config: AnyModelConfig): config is SigLIPModelConfig => {
  return config.type === 'siglip';
};

export const isFluxReduxModelConfig = (config: AnyModelConfig): config is FLUXReduxModelConfig => {
  return config.type === 'flux_redux';
};

export const isExternalApiModelConfig = (
  config: AnyModelConfigWithExternal | null | undefined
): config is ExternalApiModelConfig => {
  return !!config && (config as { format?: string }).format === 'external_api';
};

export const isUnknownModelConfig = (config: AnyModelConfig): config is UnknownModelConfig => {
  return config.type === 'unknown';
};

export const isFluxKontextModelConfig = (config: AnyModelConfig): config is FLUXKontextModelConfig => {
  return config.type === 'main' && config.base === 'flux' && config.name.toLowerCase().includes('kontext');
};

export const isNonRefinerMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base !== 'sdxl-refiner';
};

export const isMainOrExternalModelConfig = (
  config: AnyModelConfigWithExternal
): config is MainOrExternalModelConfig => {
  if (isExternalApiModelConfig(config)) {
    return true;
  }
  return isNonRefinerMainModelConfig(config);
};

export const isRefinerMainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'sdxl-refiner';
};

const isFluxDevMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'flux' && config.variant === 'dev';
};

const isFlux2Klein9BMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'flux2' && config.name.toLowerCase().includes('9b');
};

const isFlux2DevMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'flux2' && config.variant === 'dev';
};

export const isFlux2DevDiffusersMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return isFlux2DevMainModelConfig(config) && config.format === 'diffusers';
};

const isIdeogram4MainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'ideogram-4';
};

export const isNonCommercialMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return (
    isFluxDevMainModelConfig(config) ||
    isFlux2Klein9BMainModelConfig(config) ||
    isFlux2DevMainModelConfig(config) ||
    isIdeogram4MainModelConfig(config)
  );
};

export const isFluxFillMainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'flux' && config.variant === 'dev_fill';
};

/**
 * The submodels an SDNQ pipeline install must expose before it can act as a component source.
 * Mirrors `_REQUIRED_PIPELINE_SUBMODELS` / `is_self_contained_sdnq_pipeline()` in
 * `invokeai/app/invocations/model.py` — the frontend and the backend must agree on what "complete"
 * means, or the graph builders offer a source the invocation validation then rejects.
 */
const SDNQ_PIPELINE_REQUIRED_SUBMODELS = ['transformer', 'vae', 'text_encoder', 'tokenizer'] as const;

/**
 * True if an SDNQ pipeline config ships every component its loaders read from a fixed subfolder.
 *
 * A truthy `submodels` map is not enough: a partial pipeline can expose only the transformer, and a
 * malformed model_index.json can expose the components while omitting the transformer. Either would
 * otherwise be offered in the source pickers, auto-selected by the graph builders, and only rejected
 * by the backend after graph construction.
 */
export const isSelfContainedSDNQPipeline = (config: AnyModelConfig): boolean => {
  return hasSubmodels(config, SDNQ_PIPELINE_REQUIRED_SUBMODELS);
};

/**
 * FLUX.1 drives two text encoders, so a pipeline install can only replace the standalone components
 * if it also ships the T5 pair on top of the CLIP one. Mirrors
 * `_REQUIRED_FLUX1_PIPELINE_SUBMODELS` / `is_self_contained_sdnq_flux1_pipeline()` in
 * `invokeai/app/invocations/model.py`; if the two disagree, the UI either blocks a model the node
 * would have accepted or builds a graph the node then rejects.
 */
const SDNQ_FLUX1_PIPELINE_REQUIRED_SUBMODELS = [
  ...SDNQ_PIPELINE_REQUIRED_SUBMODELS,
  'text_encoder_2',
  'tokenizer_2',
] as const;

const hasSubmodels = (config: AnyModelConfig, required: readonly string[]): boolean => {
  const submodels = (config as { submodels?: unknown }).submodels;
  if (typeof submodels !== 'object' || submodels === null) {
    return false;
  }
  return required.every((submodel) => Boolean((submodels as Record<string, unknown>)[submodel]));
};

/**
 * True for a FLUX.1 SDNQ pipeline that ships every component the FLUX graph needs, so the
 * standalone T5 / CLIP / VAE selections are not required.
 */
export const isSelfContainedSDNQFlux1Pipeline = (config: AnyModelConfig): boolean => {
  if ((config as { format?: unknown }).format !== 'sdnq_quantized') {
    return false;
  }
  return hasSubmodels(config, SDNQ_FLUX1_PIPELINE_REQUIRED_SUBMODELS);
};

export const isZImageDiffusersMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  if (config.type !== 'main' || config.base !== 'z-image') {
    return false;
  }
  // Read `format` and `submodels` as plain strings/unknown so TS doesn't narrow away the
  // `sdnq_quantized` branch. The OpenAPI schema is regenerated separately and currently
  // doesn't list the `sdnq_quantized` Z-Image format variant.
  const format = (config as { format?: unknown }).format as string | undefined;
  if (format === 'diffusers') {
    return true;
  }
  // SDNQ-quantized ZImagePipeline folders carry the same submodels layout (transformer, vae,
  // text_encoder, ...) as a plain diffusers ZImagePipeline. Single-file SDNQ Z-Image
  // checkpoints have no submodels and must not match here, and neither may a partial pipeline.
  if (format !== 'sdnq_quantized') {
    return false;
  }
  return isSelfContainedSDNQPipeline(config);
};

export const isFlux2DiffusersMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  if (config.type !== 'main' || config.base !== 'flux2') {
    return false;
  }
  // Same reasoning as isZImageDiffusersMainModelConfig: an SDNQ FLUX.2 pipeline folder ships
  // the same submodels (transformer/text_encoder/tokenizer/vae) and qualifies as a source model.
  const format = (config as { format?: unknown }).format as string | undefined;
  if (format === 'diffusers') {
    return true;
  }
  if (format !== 'sdnq_quantized') {
    return false;
  }
  return isSelfContainedSDNQPipeline(config);
};

export const isQwenImageDiffusersMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'qwen-image' && config.format === 'diffusers';
};

export const isWanDiffusersMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'wan' && config.format === 'diffusers';
};

/** The single-file Wan main formats. Both are transformer-only: one file holds one
 *  A14B expert, and the VAE + UMT5-XXL encoder have to come from somewhere else.
 *  Anything gating on that property must use this, not a bare `=== 'gguf_quantized'`
 *  — the two formats are interchangeable here and drifting apart has bitten us. */
const WAN_SINGLE_FILE_FORMATS = ['gguf_quantized', 'checkpoint'] as const;

export const isWanSingleFileMainModelConfig = (config: AnyModelConfigWithExternal): config is MainModelConfig => {
  // Takes AnyModelConfig, not a structural `{base?; type?; format?}`. An all-optional
  // parameter type is a *weak type*, which TypeScript satisfies with any object sharing
  // one property name — so `ModelIdentifierField` (base + type, no format) would compile
  // and silently return false, disabling every gate below it. The bare
  // `format === 'gguf_quantized'` this replaced was at least a compile error there.
  return (
    config.type === 'main' &&
    config.base === 'wan' &&
    (WAN_SINGLE_FILE_FORMATS as readonly string[]).includes(config.format)
  );
};

/** TI2V-5B is the single-transformer Wan variant: it has no expert pair, so no expert
 *  tag on it means anything. Both predicates below have to agree about that, or a file
 *  can fall through the gap between them. */
const isWanTi2v5bConfig = (config: AnyModelConfigWithExternal): boolean =>
  'variant' in config && config.variant === 'ti2v_5b';

/** Wan single-file main models *tagged* as the low-noise expert. This is the narrow,
 *  tag-based test, and its only job is deciding what to hide from the primary main
 *  dropdown — see `selectPrimaryMainModelOptions`, its one caller. Deliberately not
 *  exported: the Transformer (Low Noise) picker needs the wider test below, and reaching
 *  for this one there is the mistake that left untagged pairs unwireable.
 *
 *  TI2V-5B is excluded for the same reason it is excluded from the partner picker. The
 *  two exclusions have to match: hiding a 5B from the primary list steers it toward a
 *  partner slot that will not offer it either, which is how a model ends up reachable
 *  from nowhere. Such a record is not hypothetical — the pre-branch GGUF probe applied
 *  the tag without consulting the variant, so a 5B named `...-low_noise.gguf` installed
 *  before this branch still carries `expert='low'` today. */
const isWanSingleFileLowNoiseMainModelConfig = (config: AnyModelConfigWithExternal): config is MainModelConfig => {
  return (
    isWanSingleFileMainModelConfig(config) &&
    !isWanTi2v5bConfig(config) &&
    'expert' in config &&
    config.expert === 'low'
  );
};

/** What the Transformer (Low Noise) picker may offer.
 *
 *  Deliberately wider than the tag test above. Since #9505 the *wiring* decides which
 *  expert a file is used as and the `expert` tag is only advisory, so requiring
 *  `expert === 'low'` here strands every pair that probes to `none`/`none`: both halves
 *  show up in the primary picker (which hides only models tagged `low`) and neither
 *  shows up here, leaving the pair impossible to assemble outside the workflow editor.
 *  That is the exact case this branch exists to support. It is also not something the
 *  user can tag their way out of: `expert` is absent from `ModelRecordChanges`, so no
 *  edit sets it. Re-probing via Reidentify recomputes it, but only from the filename,
 *  which for an untagged file returns `none` again.
 *
 *  Two exclusions. Files tagged `high` belong in the primary slot — the loader would
 *  only swap them back. TI2V-5B is single-transformer, so it has no partner at all and
 *  offering one could only produce the variant mismatch the loader rejects. */
export const isWanLowNoisePartnerOption = (config: AnyModelConfigWithExternal): config is MainModelConfig => {
  if (!isWanSingleFileMainModelConfig(config)) {
    return false;
  }
  if ('expert' in config && config.expert === 'high') {
    return false;
  }
  return !isWanTi2v5bConfig(config);
};

/** Narrows a main-model list to what may be offered as the *primary* main. Every list
 *  the user can pick a primary main from must go through this — there are three
 *  (MainModelPicker, InitialStateMainModelPicker, and the auto-select in the
 *  modelsLoaded listener), and filtering in only some of them means the excluded models
 *  are still reachable.
 *
 *  It only hides Wan A14B low-noise experts, and only when the user has a partner to
 *  pick instead. The steer is worth making — a low-noise expert belongs in the
 *  Transformer (Low Noise) slot, and running it alone gives visibly worse output — but
 *  since #9505 the loader accepts an unpaired low expert with a warning rather than
 *  refusing it. Hiding unconditionally would leave someone whose only Wan file is a low
 *  expert staring at a list that doesn't contain their model, with nothing to do about
 *  it. Partner-aware, the list degrades instead of dead-ending.
 *
 *  A partner is another single-file Wan main of the same variant that isn't itself
 *  tagged low — i.e. the high-noise or untagged half of the same pair. */
export const selectPrimaryMainModelOptions = <T extends AnyModelConfigWithExternal>(configs: T[]): T[] => {
  // Annotated `: boolean` rather than left as an inferred type predicate. Two predicates
  // narrowing to the same type would make the negated one resolve `candidate` to `never`
  // below, and the `variant` read would stop compiling.
  const isLowExpert = (config: T): boolean => isWanSingleFileLowNoiseMainModelConfig(config);
  const variantOf = (config: T): string | null =>
    'variant' in config && typeof config.variant === 'string' ? config.variant : null;

  const hasPartner = (low: T): boolean =>
    configs.some(
      (candidate) =>
        candidate.key !== low.key &&
        isWanSingleFileMainModelConfig(candidate) &&
        !isLowExpert(candidate) &&
        variantOf(candidate) === variantOf(low)
    );

  return configs.filter((config) => !isLowExpert(config) || !hasPartner(config));
};

export const isWanLoRAModelConfig = (config: AnyModelConfig): config is WanLoRAModelConfig => {
  return config.type === 'lora' && config.base === 'wan';
};

export const isTIModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'embedding';
};

type ExternalModelInstallSource = {
  type: 'external';
  provider_id: string;
  provider_model_id: string;
};
type ModelInstallSource = S['ModelInstallJob']['source'] | ExternalModelInstallSource;
export type ModelInstallJob = Omit<S['ModelInstallJob'], 'source'> & {
  source: ModelInstallSource;
};
export type ModelInstallStatus = S['InstallStatus'];

// Graphs
export type Graph = S['Graph'];
export type NonNullableGraph = SetRequired<Graph, 'nodes' | 'edges'>;
export type Batch = S['Batch'];
export const zWorkflowRecordOrderBy = z.enum(['name', 'created_at', 'updated_at', 'opened_at', 'is_public']);
export type WorkflowRecordOrderBy = z.infer<typeof zWorkflowRecordOrderBy>;
assert<Equals<S['WorkflowRecordOrderBy'], WorkflowRecordOrderBy>>();

export const zSQLiteDirection = z.enum(['ASC', 'DESC']);
export type SQLiteDirection = z.infer<typeof zSQLiteDirection>;
assert<Equals<S['SQLiteDirection'], SQLiteDirection>>();
export type WorkflowRecordListItemWithThumbnailDTO = S['WorkflowRecordListItemWithThumbnailDTO'];

type KeysOfUnion<T> = T extends T ? keyof T : never;

export type AnyInvocation = Exclude<
  NonNullable<S['Graph']['nodes']>[string],
  S['CoreMetadataInvocation'] | S['MetadataInvocation'] | S['MetadataItemInvocation'] | S['MergeMetadataInvocation']
>;
export type AnyInvocationIncMetadata = NonNullable<S['Graph']['nodes']>[string];

export type InvocationType = AnyInvocation['type'];
type InvocationOutputMap = S['InvocationOutputMap'];
export type AnyInvocationOutput = InvocationOutputMap[InvocationType];

export type Invocation<T extends InvocationType> = Extract<AnyInvocation, { type: T }>;
// export type InvocationOutput<T extends InvocationType> = InvocationOutputMap[T];

type NonInputFields = 'id' | 'type' | 'is_intermediate' | 'use_cache' | 'board' | 'metadata';
export type AnyInvocationInputField = Exclude<KeysOfUnion<Required<AnyInvocation>>, NonInputFields>;
export type InputFields<T extends AnyInvocation> = Extract<keyof T, AnyInvocationInputField>;

type ExcludeIndexSignature<T> = {
  [K in keyof T as string extends K ? never : K]: T[K];
};

export type CoreMetadataFields = Exclude<
  keyof ExcludeIndexSignature<components['schemas']['CoreMetadataInvocation']>,
  NonInputFields
>;

type NonOutputFields = 'type';
export type AnyInvocationOutputField = Exclude<KeysOfUnion<Required<AnyInvocationOutput>>, NonOutputFields>;
export type OutputFields<T extends AnyInvocation> = Extract<
  keyof InvocationOutputMap[T['type']],
  AnyInvocationOutputField
>;

// Node Outputs
export type ImageOutput = S['ImageOutput'];

export type BoardRecordOrderBy = S['BoardRecordOrderBy'];
export type StarterModel = S['StarterModel'];

export type GetHFTokenStatusResponse =
  paths['/api/v2/models/hf_login']['get']['responses']['200']['content']['application/json'];
export type SetHFTokenResponse = NonNullable<
  paths['/api/v2/models/hf_login']['post']['responses']['200']['content']['application/json']
>;
export type ResetHFTokenResponse = NonNullable<
  paths['/api/v2/models/hf_login']['delete']['responses']['200']['content']['application/json']
>;
export type SetHFTokenArg = NonNullable<
  paths['/api/v2/models/hf_login']['post']['requestBody']['content']['application/json']
>;

export type UploadImageArg = {
  /**
   * The file object to upload
   */
  file: File;
  /**
   * THe category of image to upload
   */
  image_category: ImageCategory;
  /**
   * Whether the uploaded image is an intermediate image (intermediate images are not shown int he gallery)
   */
  is_intermediate: boolean;
  /**
   * The session with which to associate the uploaded image
   */
  session_id?: string;
  /**
   * The board id to add the image to
   */
  board_id?: string;
  /**
   * Whether or not to crop the image to its bounding box before saving
   */
  crop_visible?: boolean;
  /**
   * Metadata to embed in the image when saving it
   */
  metadata?: JsonObject;
  /**
   * Whether this upload should be "silent" (no toast on upload, no changing of gallery view)
   */
  silent?: boolean;
  /**
   * Whether this is the first upload of a batch (used when displaying user feedback with toasts - ignored if the upload is silent)
   */
  isFirstUploadOfBatch?: boolean;
  /**
   * If provided, the uploaded image will resized to the given dimensions.
   */
  resize_to?: Dimensions;
};

export type ImageUploadEntryResponse = S['ImageUploadEntry'];
export type ImageUploadEntryRequest = paths['/api/v1/images/']['post']['requestBody']['content']['application/json'];

// Videos
export type VideoDTO = S['VideoDTO'];
/** @knipignore Used by Phase 4+ video gallery mutations. */
export type VideoRecordChanges = S['VideoRecordChanges'];
export type OffsetPaginatedResults_VideoDTO_ = S['OffsetPaginatedResults_VideoDTO_'];
export type ListVideosArgs = NonNullable<paths['/api/v1/videos/']['get']['parameters']['query']>;
export type ListVideosResponse = paths['/api/v1/videos/']['get']['responses']['200']['content']['application/json'];
export type GetVideoNamesArgs = NonNullable<paths['/api/v1/videos/names']['get']['parameters']['query']>;
export type GetVideoNamesResult =
  paths['/api/v1/videos/names']['get']['responses']['200']['content']['application/json'];

export type UploadVideoArg = {
  /** The MP4 (or other accepted video) file to upload. */
  file: File;
  /** The category of video to upload. Reuses the image category enum. */
  video_category: ImageCategory;
  /** Whether the uploaded video is an intermediate (intermediates are not shown in the gallery). */
  is_intermediate: boolean;
  /** The session with which to associate the uploaded video, if any. */
  session_id?: string;
  /** The board to add the video to, if any. */
  board_id?: string;
  /** Metadata JSON to attach to the video record. */
  metadata?: JsonObject;
  /** Suppress the upload toast / gallery navigation side effects. */
  silent?: boolean;
  /** Whether this is the first upload of a batch (used by toast logic). */
  isFirstUploadOfBatch?: boolean;
};

// Polymorphic gallery items (images + videos). Consumed by the gallery wiring in Phase 4.
/** @knipignore Consumed by gallery wiring in Phase 4. */
export type GalleryItem = S['GalleryItem'];
/** @knipignore Consumed by gallery wiring in Phase 4. */
export type GalleryItemKind = S['GalleryItemKind'];
/** @knipignore Consumed by gallery wiring in Phase 4. */
export type GalleryItemRef = S['GalleryItemRef'];
/** @knipignore Consumed by gallery wiring in Phase 4. */
export type GalleryItemNamesResult = S['GalleryItemNamesResult'];
/** @knipignore Consumed by gallery wiring in Phase 4. */
export type OffsetPaginatedResults_GalleryItem_ = S['OffsetPaginatedResults_GalleryItem_'];
export type ListGalleryItemsArgs = NonNullable<paths['/api/v1/gallery/items/']['get']['parameters']['query']>;
export type ListGalleryItemsResponse =
  paths['/api/v1/gallery/items/']['get']['responses']['200']['content']['application/json'];
export type ListGalleryItemNamesArgs = NonNullable<paths['/api/v1/gallery/item_names']['get']['parameters']['query']>;
export type ListGalleryItemNamesResult =
  paths['/api/v1/gallery/item_names']['get']['responses']['200']['content']['application/json'];
