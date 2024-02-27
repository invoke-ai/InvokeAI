import type { UseToastOptions } from '@invoke-ai/ui-library';
import type { EntityState } from '@reduxjs/toolkit';
import type { components, paths } from 'services/api/schema';
import type { O } from 'ts-toolbelt';

export type S = components['schemas'];

export type ImageCache = EntityState<ImageDTO, string>;

export type ListImagesArgs = NonNullable<paths['/api/v1/images/']['get']['parameters']['query']>;

export type DeleteBoardResult =
  paths['/api/v1/boards/{board_id}']['delete']['responses']['200']['content']['application/json'];

export type ListBoardsArg = NonNullable<paths['/api/v1/boards/']['get']['parameters']['query']>;

export type UpdateBoardArg = paths['/api/v1/boards/{board_id}']['patch']['parameters']['path'] & {
  changes: paths['/api/v1/boards/{board_id}']['patch']['requestBody']['content']['application/json'];
};

export type BatchConfig =
  paths['/api/v1/queue/{queue_id}/enqueue_batch']['post']['requestBody']['content']['application/json'];

export type EnqueueBatchResult = components['schemas']['EnqueueBatchResult'];

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
export type BoardChanges = S['BoardChanges'];
export type ImageChanges = S['ImageRecordChanges'];
export type ImageCategory = S['ImageCategory'];
export type ResourceOrigin = S['ResourceOrigin'];
export type ImageField = S['ImageField'];
export type OffsetPaginatedResults_BoardDTO_ = S['OffsetPaginatedResults_BoardDTO_'];
export type OffsetPaginatedResults_ImageDTO_ = S['OffsetPaginatedResults_ImageDTO_'];

// Models
export type ModelType = S['ModelType'];
export type SubModelType = S['SubModelType'];
export type BaseModelType = S['BaseModelType'];
export type MainModelField = S['MainModelField'];
export type VAEModelField = S['VAEModelField'];
export type LoRAModelField = S['LoRAModelField'];
export type LoRAModelFormat = S['LoRAModelFormat'];
export type ControlNetModelField = S['ControlNetModelField'];
export type IPAdapterModelField = S['IPAdapterModelField'];
export type T2IAdapterModelField = S['T2IAdapterModelField'];
export type ControlField = S['ControlField'];
export type IPAdapterField = S['IPAdapterField'];

// Model Configs

// TODO(MM2): Can we make key required in the pydantic model?
export type LoRAModelConfig = S['LoRAConfig'];
// TODO(MM2): Can we rename this from Vae -> VAE
export type VAEModelConfig = S['VaeCheckpointConfig'] | S['VaeDiffusersConfig'];
export type ControlNetModelConfig = S['ControlNetDiffusersConfig'] | S['ControlNetCheckpointConfig'];
export type IPAdapterModelConfig = S['IPAdapterConfig'];
// TODO(MM2): Can we rename this to T2IAdapterConfig
export type T2IAdapterModelConfig = S['T2IConfig'];
export type TextualInversionModelConfig = S['TextualInversionConfig'];
export type DiffusersModelConfig = S['MainDiffusersConfig'];
export type CheckpointModelConfig = S['MainCheckpointConfig'];
type CLIPVisionDiffusersConfig = S['CLIPVisionDiffusersConfig'];
export type MainModelConfig = DiffusersModelConfig | CheckpointModelConfig;
export type RefinerMainModelConfig = Omit<MainModelConfig, 'base'> & { base: 'sdxl-refiner' };
export type NonRefinerMainModelConfig = Omit<MainModelConfig, 'base'> & { base: 'any' | 'sd-1' | 'sd-2' | 'sdxl' };
export type AnyModelConfig =
  | LoRAModelConfig
  | VAEModelConfig
  | ControlNetModelConfig
  | IPAdapterModelConfig
  | T2IAdapterModelConfig
  | TextualInversionModelConfig
  | RefinerMainModelConfig
  | NonRefinerMainModelConfig
  | CLIPVisionDiffusersConfig;

type AnyModelConfig2 =
  | (S['MainDiffusersConfig'] | S['MainCheckpointConfig'])
  | (S['VaeDiffusersConfig'] | S['VaeCheckpointConfig'])
  | (S['ControlNetDiffusersConfig'] | S['ControlNetCheckpointConfig'])
  | S['LoRAConfig']
  | S['TextualInversionConfig']
  | S['IPAdapterConfig']
  | S['CLIPVisionDiffusersConfig']
  | S['T2IConfig'];

export const isLoRAModelConfig = (config: AnyModelConfig): config is LoRAModelConfig => {
  return config.type === 'lora';
};

export const isVAEModelConfig = (config: AnyModelConfig): config is VAEModelConfig => {
  return config.type === 'vae';
};

export const isControlNetModelConfig = (config: AnyModelConfig): config is ControlNetModelConfig => {
  return config.type === 'controlnet';
};

export const isIPAdapterModelConfig = (config: AnyModelConfig): config is IPAdapterModelConfig => {
  return config.type === 'ip_adapter';
};

export const isT2IAdapterModelConfig = (config: AnyModelConfig): config is T2IAdapterModelConfig => {
  return config.type === 't2i_adapter';
};

export const isTextualInversionModelConfig = (config: AnyModelConfig): config is TextualInversionModelConfig => {
  return config.type === 'embedding';
};

export const isNonRefinerMainModelConfig = (config: AnyModelConfig): config is NonRefinerMainModelConfig => {
  return config.type === 'main' && config.base !== 'sdxl-refiner';
};

export const isRefinerMainModelModelConfig = (config: AnyModelConfig): config is RefinerMainModelConfig => {
  return config.type === 'main' && config.base === 'sdxl-refiner';
};

export type MergeModelConfig = S['Body_merge'];
export type ImportModelConfig = S['Body_import_model'];
export type ModelInstallJob = S['ModelInstallJob'];
export type ModelInstallStatus = S['InstallStatus'];

export type HFModelSource = S['HFModelSource'];
export type CivitaiModelSource = S['CivitaiModelSource'];
export type LocalModelSource = S['LocalModelSource'];
export type URLModelSource = S['URLModelSource'];

// Graphs
export type Graph = S['Graph'];
export type NonNullableGraph = O.Required<Graph, 'nodes' | 'edges'>;
export type Edge = S['Edge'];
export type GraphExecutionState = S['GraphExecutionState'];
export type Batch = S['Batch'];
export type SessionQueueItemDTO = S['SessionQueueItemDTO'];
export type SessionQueueItem = S['SessionQueueItem'];
export type WorkflowRecordOrderBy = S['WorkflowRecordOrderBy'];
export type SQLiteDirection = S['SQLiteDirection'];
export type WorkflowDTO = S['WorkflowRecordDTO'];
export type WorkflowRecordListItemDTO = S['WorkflowRecordListItemDTO'];

// General nodes
export type CollectInvocation = S['CollectInvocation'];
export type IterateInvocation = S['IterateInvocation'];
export type RangeInvocation = S['RangeInvocation'];
export type RandomRangeInvocation = S['RandomRangeInvocation'];
export type RangeOfSizeInvocation = S['RangeOfSizeInvocation'];
export type ImageResizeInvocation = S['ImageResizeInvocation'];
export type ImageBlurInvocation = S['ImageBlurInvocation'];
export type ImageScaleInvocation = S['ImageScaleInvocation'];
export type InfillPatchMatchInvocation = S['InfillPatchMatchInvocation'];
export type InfillTileInvocation = S['InfillTileInvocation'];
export type CreateDenoiseMaskInvocation = S['CreateDenoiseMaskInvocation'];
export type CreateGradientMaskInvocation = S['CreateGradientMaskInvocation'];
export type CanvasPasteBackInvocation = S['CanvasPasteBackInvocation'];
export type MaskEdgeInvocation = S['MaskEdgeInvocation'];
export type RandomIntInvocation = S['RandomIntInvocation'];
export type CompelInvocation = S['CompelInvocation'];
export type DynamicPromptInvocation = S['DynamicPromptInvocation'];
export type NoiseInvocation = S['NoiseInvocation'];
export type DenoiseLatentsInvocation = S['DenoiseLatentsInvocation'];
export type SDXLLoraLoaderInvocation = S['SDXLLoraLoaderInvocation'];
export type ImageToLatentsInvocation = S['ImageToLatentsInvocation'];
export type LatentsToImageInvocation = S['LatentsToImageInvocation'];
export type ImageCollectionInvocation = S['ImageCollectionInvocation'];
export type MainModelLoaderInvocation = S['MainModelLoaderInvocation'];
export type LoraLoaderInvocation = S['LoraLoaderInvocation'];
export type ESRGANInvocation = S['ESRGANInvocation'];
export type DivideInvocation = S['DivideInvocation'];
export type ImageNSFWBlurInvocation = S['ImageNSFWBlurInvocation'];
export type ImageWatermarkInvocation = S['ImageWatermarkInvocation'];
export type SeamlessModeInvocation = S['SeamlessModeInvocation'];
export type MetadataInvocation = S['MetadataInvocation'];
export type CoreMetadataInvocation = S['CoreMetadataInvocation'];
export type MetadataItemInvocation = S['MetadataItemInvocation'];
export type MergeMetadataInvocation = S['MergeMetadataInvocation'];
export type IPAdapterMetadataField = S['IPAdapterMetadataField'];
export type T2IAdapterField = S['T2IAdapterField'];
export type LoRAMetadataField = S['LoRAMetadataField'];

// ControlNet Nodes
export type ControlNetInvocation = S['ControlNetInvocation'];
export type T2IAdapterInvocation = S['T2IAdapterInvocation'];
export type IPAdapterInvocation = S['IPAdapterInvocation'];
export type CannyImageProcessorInvocation = S['CannyImageProcessorInvocation'];
export type ColorMapImageProcessorInvocation = S['ColorMapImageProcessorInvocation'];
export type ContentShuffleImageProcessorInvocation = S['ContentShuffleImageProcessorInvocation'];
export type DepthAnythingImageProcessorInvocation = S['DepthAnythingImageProcessorInvocation'];
export type HedImageProcessorInvocation = S['HedImageProcessorInvocation'];
export type LineartAnimeImageProcessorInvocation = S['LineartAnimeImageProcessorInvocation'];
export type LineartImageProcessorInvocation = S['LineartImageProcessorInvocation'];
export type MediapipeFaceProcessorInvocation = S['MediapipeFaceProcessorInvocation'];
export type MidasDepthImageProcessorInvocation = S['MidasDepthImageProcessorInvocation'];
export type MlsdImageProcessorInvocation = S['MlsdImageProcessorInvocation'];
export type NormalbaeImageProcessorInvocation = S['NormalbaeImageProcessorInvocation'];
export type DWOpenposeImageProcessorInvocation = S['DWOpenposeImageProcessorInvocation'];
export type PidiImageProcessorInvocation = S['PidiImageProcessorInvocation'];
export type ZoeDepthImageProcessorInvocation = S['ZoeDepthImageProcessorInvocation'];

// Node Outputs
export type ImageOutput = S['ImageOutput'];
export type StringOutput = S['StringOutput'];
export type FloatOutput = S['FloatOutput'];
export type IntegerOutput = S['IntegerOutput'];
export type IterateInvocationOutput = S['IterateInvocationOutput'];
export type CollectInvocationOutput = S['CollectInvocationOutput'];
export type LatentsOutput = S['LatentsOutput'];

// Post-image upload actions, controls workflows when images are uploaded

export type ControlAdapterAction = {
  type: 'SET_CONTROL_ADAPTER_IMAGE';
  id: string;
};

export type InitialImageAction = {
  type: 'SET_INITIAL_IMAGE';
};

export type NodesAction = {
  type: 'SET_NODES_IMAGE';
  nodeId: string;
  fieldName: string;
};

export type CanvasInitialImageAction = {
  type: 'SET_CANVAS_INITIAL_IMAGE';
};

export type ToastAction = {
  type: 'TOAST';
  toastOptions?: UseToastOptions;
};

export type AddToBatchAction = {
  type: 'ADD_TO_BATCH';
};

export type PostUploadAction =
  | ControlAdapterAction
  | InitialImageAction
  | NodesAction
  | CanvasInitialImageAction
  | ToastAction
  | AddToBatchAction;

type TypeGuard<T> = {
  (input: unknown): input is T;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type TypeGuardFor<T extends TypeGuard<any>> = T extends TypeGuard<infer U> ? U : never;
