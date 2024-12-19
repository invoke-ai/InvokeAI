import type { components, paths } from 'services/api/schema';
import type { JsonObject, SetRequired } from 'type-fest';

export type S = components['schemas'];

export type ListImagesArgs = NonNullable<paths['/api/v1/images/']['get']['parameters']['query']>;
export type ListImagesResponse = paths['/api/v1/images/']['get']['responses']['200']['content']['application/json'];

export type ListBoardsArgs = NonNullable<paths['/api/v1/boards/']['get']['parameters']['query']>;

export type DeleteBoardResult =
  paths['/api/v1/boards/{board_id}']['delete']['responses']['200']['content']['application/json'];

export type CreateBoardArg = paths['/api/v1/boards/']['post']['parameters']['query'];

export type UpdateBoardArg = paths['/api/v1/boards/{board_id}']['patch']['parameters']['path'] & {
  changes: paths['/api/v1/boards/{board_id}']['patch']['requestBody']['content']['application/json'];
};

export type GraphAndWorkflowResponse =
  paths['/api/v1/images/i/{image_name}/workflow']['get']['responses']['200']['content']['application/json'];

export type BatchConfig =
  paths['/api/v1/queue/{queue_id}/enqueue_batch']['post']['requestBody']['content']['application/json'];

export type InputFieldJSONSchemaExtra = S['InputFieldJSONSchemaExtra'];
export type OutputFieldJSONSchemaExtra = S['OutputFieldJSONSchemaExtra'];
export type InvocationJSONSchemaExtra = S['UIConfigBase'];

// App Info
export type AppVersion = S['AppVersion'];
export type AppConfig = S['AppConfig'];
export type AppDependencyVersions = S['AppDependencyVersions'];

// Images
export type ImageDTO = S['ImageDTO'];
export type BoardDTO = S['BoardDTO'];
export type ImageCategory = S['ImageCategory'];
export type OffsetPaginatedResults_ImageDTO_ = S['OffsetPaginatedResults_ImageDTO_'];

// Models
export type ModelType = S['ModelType'];
export type BaseModelType = S['BaseModelType'];

// Model Configs

export type ControlLoRAModelConfig = S['ControlLoRALyCORISConfig'] | S['ControlLoRADiffusersConfig'];
// TODO(MM2): Can we make key required in the pydantic model?
export type LoRAModelConfig = S['LoRADiffusersConfig'] | S['LoRALyCORISConfig'];
// TODO(MM2): Can we rename this from Vae -> VAE
export type VAEModelConfig = S['VAECheckpointConfig'] | S['VAEDiffusersConfig'];
export type ControlNetModelConfig = S['ControlNetDiffusersConfig'] | S['ControlNetCheckpointConfig'];
export type IPAdapterModelConfig = S['IPAdapterInvokeAIConfig'] | S['IPAdapterCheckpointConfig'];
export type T2IAdapterModelConfig = S['T2IAdapterConfig'];
export type CLIPEmbedModelConfig = S['CLIPEmbedDiffusersConfig'];
export type CLIPLEmbedModelConfig = S['CLIPLEmbedDiffusersConfig'];
export type CLIPGEmbedModelConfig = S['CLIPGEmbedDiffusersConfig'];
export type T5EncoderModelConfig = S['T5EncoderConfig'];
export type T5EncoderBnbQuantizedLlmInt8bModelConfig = S['T5EncoderBnbQuantizedLlmInt8bConfig'];
export type SpandrelImageToImageModelConfig = S['SpandrelImageToImageConfig'];
type TextualInversionModelConfig = S['TextualInversionFileConfig'] | S['TextualInversionFolderConfig'];
type DiffusersModelConfig = S['MainDiffusersConfig'];
export type CheckpointModelConfig = S['MainCheckpointConfig'];
type CLIPVisionDiffusersConfig = S['CLIPVisionDiffusersConfig'];
export type MainModelConfig = DiffusersModelConfig | CheckpointModelConfig;
export type AnyModelConfig =
  | ControlLoRAModelConfig
  | LoRAModelConfig
  | VAEModelConfig
  | ControlNetModelConfig
  | IPAdapterModelConfig
  | T5EncoderModelConfig
  | T5EncoderBnbQuantizedLlmInt8bModelConfig
  | CLIPEmbedModelConfig
  | T2IAdapterModelConfig
  | SpandrelImageToImageModelConfig
  | TextualInversionModelConfig
  | MainModelConfig
  | CLIPVisionDiffusersConfig;

/**
 * Checks if a list of submodels contains any that match a given variant or type
 * @param submodels The list of submodels to check
 * @param checkStr The string to check against for variant or type
 * @returns A boolean
 */
const checkSubmodel = (submodels: AnyModelConfig['submodels'], checkStr: string): boolean => {
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

export const isVAEModelConfig = (config: AnyModelConfig, excludeSubmodels?: boolean): config is VAEModelConfig => {
  return config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config));
};

export const isNonFluxVAEModelConfig = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is VAEModelConfig => {
  return (
    (config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config))) &&
    config.base !== 'flux'
  );
};

export const isFluxVAEModelConfig = (config: AnyModelConfig, excludeSubmodels?: boolean): config is VAEModelConfig => {
  return (
    (config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config))) &&
    config.base === 'flux'
  );
};

export const isControlNetModelConfig = (config: AnyModelConfig): config is ControlNetModelConfig => {
  return config.type === 'controlnet';
};

export const isControlLayerModelConfig = (
  config: AnyModelConfig
): config is ControlNetModelConfig | T2IAdapterModelConfig | ControlLoRAModelConfig => {
  return config.type === 'controlnet' || config.type === 't2i_adapter' || config.type === 'control_lora';
};

export const isIPAdapterModelConfig = (config: AnyModelConfig): config is IPAdapterModelConfig => {
  return config.type === 'ip_adapter';
};

export const isCLIPVisionModelConfig = (config: AnyModelConfig): config is CLIPVisionDiffusersConfig => {
  return config.type === 'clip_vision';
};

export const isT2IAdapterModelConfig = (config: AnyModelConfig): config is T2IAdapterModelConfig => {
  return config.type === 't2i_adapter';
};

export const isT5EncoderModelConfig = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is T5EncoderModelConfig | T5EncoderBnbQuantizedLlmInt8bModelConfig => {
  return (
    config.type === 't5_encoder' ||
    (!excludeSubmodels && config.type === 'main' && checkSubmodels(['t5_encoder'], config))
  );
};

export const isCLIPEmbedModelConfig = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is CLIPEmbedModelConfig => {
  return (
    config.type === 'clip_embed' ||
    (!excludeSubmodels && config.type === 'main' && checkSubmodels(['clip_embed'], config))
  );
};

export const isCLIPLEmbedModelConfig = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is CLIPLEmbedModelConfig => {
  return (
    (config.type === 'clip_embed' && config.variant === 'large') ||
    (!excludeSubmodels && config.type === 'main' && checkSubmodels(['clip_embed', 'large'], config))
  );
};

export const isCLIPGEmbedModelConfig = (
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

export const isNonRefinerMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base !== 'sdxl-refiner';
};

export const isCheckpointMainModelConfig = (config: AnyModelConfig): config is CheckpointModelConfig => {
  return config.type === 'main' && (config.format === 'checkpoint' || config.format === 'bnb_quantized_nf4b');
};

export const isRefinerMainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'sdxl-refiner';
};

export const isSDXLMainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'sdxl';
};

export const isSD3MainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'sd-3';
};

export const isFluxMainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'flux';
};

export const isNonSDXLMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && (config.base === 'sd-1' || config.base === 'sd-2');
};

export const isTIModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'embedding';
};

export type ModelInstallJob = S['ModelInstallJob'];
export type ModelInstallStatus = S['InstallStatus'];

// Graphs
export type Graph = S['Graph'];
export type NonNullableGraph = SetRequired<Graph, 'nodes' | 'edges'>;
export type Batch = S['Batch'];
export type SessionQueueItemDTO = S['SessionQueueItemDTO'];
export type WorkflowRecordOrderBy = S['WorkflowRecordOrderBy'];
export type SQLiteDirection = S['SQLiteDirection'];
export type WorkflowRecordListItemDTO = S['WorkflowRecordListItemDTO'];

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
};
