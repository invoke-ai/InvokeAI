import type { UseToastOptions } from '@invoke-ai/ui-library';
import type { EntityState } from '@reduxjs/toolkit';
import type { components, paths } from 'services/api/schema';
import type { O } from 'ts-toolbelt';

type s = components['schemas'];

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

export type InputFieldJSONSchemaExtra = s['InputFieldJSONSchemaExtra'];
export type OutputFieldJSONSchemaExtra = s['OutputFieldJSONSchemaExtra'];
export type InvocationJSONSchemaExtra = s['UIConfigBase'];

// App Info
export type AppVersion = s['AppVersion'];
export type AppConfig = s['AppConfig'];
export type AppDependencyVersions = s['AppDependencyVersions'];

// Images
export type ImageDTO = s['ImageDTO'];
export type BoardDTO = s['BoardDTO'];
export type BoardChanges = s['BoardChanges'];
export type ImageChanges = s['ImageRecordChanges'];
export type ImageCategory = s['ImageCategory'];
export type ResourceOrigin = s['ResourceOrigin'];
export type ImageField = s['ImageField'];
export type OffsetPaginatedResults_BoardDTO_ = s['OffsetPaginatedResults_BoardDTO_'];
export type OffsetPaginatedResults_ImageDTO_ = s['OffsetPaginatedResults_ImageDTO_'];

// Models
export type ModelType = s['invokeai__backend__model_management__models__base__ModelType'];
export type SubModelType = s['SubModelType'];
export type BaseModelType = s['invokeai__backend__model_management__models__base__BaseModelType'];
export type MainModelField = s['MainModelField'];
export type VAEModelField = s['VAEModelField'];
export type LoRAModelField = s['LoRAModelField'];
export type LoRAModelFormat = s['LoRAModelFormat'];
export type ControlNetModelField = s['ControlNetModelField'];
export type IPAdapterModelField = s['IPAdapterModelField'];
export type T2IAdapterModelField = s['T2IAdapterModelField'];
export type ModelsList = s['invokeai__app__api__routers__models__ModelsList'];
export type ControlField = s['ControlField'];
export type IPAdapterField = s['IPAdapterField'];

// Model Configs
export type LoRAModelConfig = s['LoRAModelConfig'];
export type VaeModelConfig = s['VaeModelConfig'];
export type ControlNetModelCheckpointConfig = s['ControlNetModelCheckpointConfig'];
export type ControlNetModelDiffusersConfig = s['ControlNetModelDiffusersConfig'];
export type ControlNetModelConfig = ControlNetModelCheckpointConfig | ControlNetModelDiffusersConfig;
export type IPAdapterModelInvokeAIConfig = s['IPAdapterModelInvokeAIConfig'];
export type IPAdapterModelConfig = IPAdapterModelInvokeAIConfig;
export type T2IAdapterModelDiffusersConfig = s['T2IAdapterModelDiffusersConfig'];
export type T2IAdapterModelConfig = T2IAdapterModelDiffusersConfig;
export type TextualInversionModelConfig = s['TextualInversionModelConfig'];
export type DiffusersModelConfig =
  | s['StableDiffusion1ModelDiffusersConfig']
  | s['StableDiffusion2ModelDiffusersConfig']
  | s['StableDiffusionXLModelDiffusersConfig'];
export type CheckpointModelConfig =
  | s['StableDiffusion1ModelCheckpointConfig']
  | s['StableDiffusion2ModelCheckpointConfig']
  | s['StableDiffusionXLModelCheckpointConfig'];
export type MainModelConfig = DiffusersModelConfig | CheckpointModelConfig;
export type AnyModelConfig =
  | LoRAModelConfig
  | VaeModelConfig
  | ControlNetModelConfig
  | IPAdapterModelConfig
  | T2IAdapterModelConfig
  | TextualInversionModelConfig
  | MainModelConfig;

export type MergeModelConfig = s['Body_merge_models'];
export type ImportModelConfig = s['Body_import_model'];

// Graphs
export type Graph = s['Graph'];
export type NonNullableGraph = O.Required<Graph, 'nodes' | 'edges'>;
export type Edge = s['Edge'];
export type GraphExecutionState = s['GraphExecutionState'];
export type Batch = s['Batch'];
export type SessionQueueItemDTO = s['SessionQueueItemDTO'];
export type SessionQueueItem = s['SessionQueueItem'];
export type WorkflowRecordOrderBy = s['WorkflowRecordOrderBy'];
export type SQLiteDirection = s['SQLiteDirection'];
export type WorkflowDTO = s['WorkflowRecordDTO'];
export type WorkflowRecordListItemDTO = s['WorkflowRecordListItemDTO'];

// General nodes
export type CollectInvocation = s['CollectInvocation'];
export type IterateInvocation = s['IterateInvocation'];
export type RangeInvocation = s['RangeInvocation'];
export type RandomRangeInvocation = s['RandomRangeInvocation'];
export type RangeOfSizeInvocation = s['RangeOfSizeInvocation'];
export type ImageResizeInvocation = s['ImageResizeInvocation'];
export type ImageBlurInvocation = s['ImageBlurInvocation'];
export type ImageScaleInvocation = s['ImageScaleInvocation'];
export type InfillPatchMatchInvocation = s['InfillPatchMatchInvocation'];
export type InfillTileInvocation = s['InfillTileInvocation'];
export type CreateDenoiseMaskInvocation = s['CreateDenoiseMaskInvocation'];
export type MaskEdgeInvocation = s['MaskEdgeInvocation'];
export type RandomIntInvocation = s['RandomIntInvocation'];
export type CompelInvocation = s['CompelInvocation'];
export type DynamicPromptInvocation = s['DynamicPromptInvocation'];
export type NoiseInvocation = s['NoiseInvocation'];
export type DenoiseLatentsInvocation = s['DenoiseLatentsInvocation'];
export type SDXLLoraLoaderInvocation = s['SDXLLoraLoaderInvocation'];
export type ImageToLatentsInvocation = s['ImageToLatentsInvocation'];
export type LatentsToImageInvocation = s['LatentsToImageInvocation'];
export type ImageCollectionInvocation = s['ImageCollectionInvocation'];
export type MainModelLoaderInvocation = s['MainModelLoaderInvocation'];
export type LoraLoaderInvocation = s['LoraLoaderInvocation'];
export type ESRGANInvocation = s['ESRGANInvocation'];
export type DivideInvocation = s['DivideInvocation'];
export type ImageNSFWBlurInvocation = s['ImageNSFWBlurInvocation'];
export type ImageWatermarkInvocation = s['ImageWatermarkInvocation'];
export type SeamlessModeInvocation = s['SeamlessModeInvocation'];
export type LinearUIOutputInvocation = s['LinearUIOutputInvocation'];
export type MetadataInvocation = s['MetadataInvocation'];
export type CoreMetadataInvocation = s['CoreMetadataInvocation'];
export type MetadataItemInvocation = s['MetadataItemInvocation'];
export type MergeMetadataInvocation = s['MergeMetadataInvocation'];
export type IPAdapterMetadataField = s['IPAdapterMetadataField'];
export type T2IAdapterField = s['T2IAdapterField'];
export type LoRAMetadataField = s['LoRAMetadataField'];

// ControlNet Nodes
export type ControlNetInvocation = s['ControlNetInvocation'];
export type T2IAdapterInvocation = s['T2IAdapterInvocation'];
export type IPAdapterInvocation = s['IPAdapterInvocation'];
export type CannyImageProcessorInvocation = s['CannyImageProcessorInvocation'];
export type ColorMapImageProcessorInvocation = s['ColorMapImageProcessorInvocation'];
export type ContentShuffleImageProcessorInvocation = s['ContentShuffleImageProcessorInvocation'];
export type DepthAnythingImageProcessorInvocation = s['DepthAnythingImageProcessorInvocation'];
export type HedImageProcessorInvocation = s['HedImageProcessorInvocation'];
export type LineartAnimeImageProcessorInvocation = s['LineartAnimeImageProcessorInvocation'];
export type LineartImageProcessorInvocation = s['LineartImageProcessorInvocation'];
export type MediapipeFaceProcessorInvocation = s['MediapipeFaceProcessorInvocation'];
export type MidasDepthImageProcessorInvocation = s['MidasDepthImageProcessorInvocation'];
export type MlsdImageProcessorInvocation = s['MlsdImageProcessorInvocation'];
export type NormalbaeImageProcessorInvocation = s['NormalbaeImageProcessorInvocation'];
export type OpenposeImageProcessorInvocation = s['OpenposeImageProcessorInvocation'];
export type DWPoseImageProcessorInvocation = s['DWPoseImageProcessorInvocation'];
export type PidiImageProcessorInvocation = s['PidiImageProcessorInvocation'];
export type ZoeDepthImageProcessorInvocation = s['ZoeDepthImageProcessorInvocation'];

// Node Outputs
export type ImageOutput = s['ImageOutput'];
export type StringOutput = s['StringOutput'];
export type FloatOutput = s['FloatOutput'];
export type IntegerOutput = s['IntegerOutput'];
export type IterateInvocationOutput = s['IterateInvocationOutput'];
export type CollectInvocationOutput = s['CollectInvocationOutput'];
export type LatentsOutput = s['LatentsOutput'];
export type GraphInvocationOutput = s['GraphInvocationOutput'];

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
