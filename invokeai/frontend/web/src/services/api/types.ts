import { UseToastOptions } from '@chakra-ui/react';
import { EntityState } from '@reduxjs/toolkit';
import { O } from 'ts-toolbelt';
import { components, paths } from './schema';

type s = components['schemas'];

export type ImageCache = EntityState<ImageDTO>;

export type ListImagesArgs = NonNullable<
  paths['/api/v1/images/']['get']['parameters']['query']
>;

export type DeleteBoardResult =
  paths['/api/v1/boards/{board_id}']['delete']['responses']['200']['content']['application/json'];

export type ListBoardsArg = NonNullable<
  paths['/api/v1/boards/']['get']['parameters']['query']
>;

export type UpdateBoardArg =
  paths['/api/v1/boards/{board_id}']['patch']['parameters']['path'] & {
    changes: paths['/api/v1/boards/{board_id}']['patch']['requestBody']['content']['application/json'];
  };

/**
 * This is an unsafe type; the object inside is not guaranteed to be valid.
 */
export type UnsafeImageMetadata = {
  metadata: s['CoreMetadata'];
  graph: NonNullable<s['Graph']>;
};

/**
 * Marks the `type` property as required. Use for nodes.
 */
type TypeReq<T extends object> = O.Required<T, 'type'>;

export type _InputField = s['_InputField'];
export type _OutputField = s['_OutputField'];

// App Info
export type AppVersion = s['AppVersion'];
export type AppConfig = s['AppConfig'];

// Images
export type ImageDTO = s['ImageDTO'];
export type BoardDTO = s['BoardDTO'];
export type BoardChanges = s['BoardChanges'];
export type ImageChanges = s['ImageRecordChanges'];
export type ImageCategory = s['ImageCategory'];
export type ResourceOrigin = s['ResourceOrigin'];
export type ImageField = s['ImageField'];
export type ImageMetadata = s['ImageMetadata'];
export type OffsetPaginatedResults_BoardDTO_ =
  s['OffsetPaginatedResults_BoardDTO_'];
export type OffsetPaginatedResults_ImageDTO_ =
  s['OffsetPaginatedResults_ImageDTO_'];

// Models
export type ModelType = s['ModelType'];
export type SubModelType = s['SubModelType'];
export type BaseModelType = s['BaseModelType'];
export type MainModelField = s['MainModelField'];
export type OnnxModelField = s['OnnxModelField'];
export type VAEModelField = s['VAEModelField'];
export type LoRAModelField = s['LoRAModelField'];
export type ControlNetModelField = s['ControlNetModelField'];
export type ModelsList = s['ModelsList'];
export type ControlField = s['ControlField'];

// Model Configs
export type LoRAModelConfig = s['LoRAModelConfig'];
export type VaeModelConfig = s['VaeModelConfig'];
export type ControlNetModelCheckpointConfig =
  s['ControlNetModelCheckpointConfig'];
export type ControlNetModelDiffusersConfig =
  s['ControlNetModelDiffusersConfig'];
export type ControlNetModelConfig =
  | ControlNetModelCheckpointConfig
  | ControlNetModelDiffusersConfig;
export type TextualInversionModelConfig = s['TextualInversionModelConfig'];
export type DiffusersModelConfig =
  | s['StableDiffusion1ModelDiffusersConfig']
  | s['StableDiffusion2ModelDiffusersConfig']
  | s['StableDiffusionXLModelDiffusersConfig'];
export type CheckpointModelConfig =
  | s['StableDiffusion1ModelCheckpointConfig']
  | s['StableDiffusion2ModelCheckpointConfig']
  | s['StableDiffusionXLModelCheckpointConfig'];
export type OnnxModelConfig = s['ONNXStableDiffusion1ModelConfig'];
export type MainModelConfig = DiffusersModelConfig | CheckpointModelConfig;
export type AnyModelConfig =
  | LoRAModelConfig
  | VaeModelConfig
  | ControlNetModelConfig
  | TextualInversionModelConfig
  | MainModelConfig
  | OnnxModelConfig;

export type MergeModelConfig = s['Body_merge_models'];
export type ImportModelConfig = s['Body_import_model'];

// Graphs
export type Graph = s['Graph'];
export type Edge = s['Edge'];
export type GraphExecutionState = s['GraphExecutionState'];

// General nodes
export type CollectInvocation = TypeReq<s['CollectInvocation']>;
export type IterateInvocation = TypeReq<s['IterateInvocation']>;
export type RangeInvocation = TypeReq<s['RangeInvocation']>;
export type RandomRangeInvocation = TypeReq<s['RandomRangeInvocation']>;
export type RangeOfSizeInvocation = TypeReq<s['RangeOfSizeInvocation']>;
export type InpaintInvocation = TypeReq<s['InpaintInvocation']>;
export type ImageResizeInvocation = TypeReq<s['ImageResizeInvocation']>;
export type ImageScaleInvocation = TypeReq<s['ImageScaleInvocation']>;
export type RandomIntInvocation = TypeReq<s['RandomIntInvocation']>;
export type CompelInvocation = TypeReq<s['CompelInvocation']>;
export type DynamicPromptInvocation = TypeReq<s['DynamicPromptInvocation']>;
export type NoiseInvocation = TypeReq<s['NoiseInvocation']>;
export type TextToLatentsInvocation = TypeReq<s['TextToLatentsInvocation']>;
export type ONNXTextToLatentsInvocation = TypeReq<
  s['ONNXTextToLatentsInvocation']
>;
export type LatentsToLatentsInvocation = TypeReq<
  s['LatentsToLatentsInvocation']
>;
export type SDXLLoraLoaderInvocation = TypeReq<
  components['schemas']['SDXLLoraLoaderInvocation']
>;
export type ImageToLatentsInvocation = TypeReq<s['ImageToLatentsInvocation']>;
export type LatentsToImageInvocation = TypeReq<s['LatentsToImageInvocation']>;
export type ImageCollectionInvocation = TypeReq<s['ImageCollectionInvocation']>;
export type MainModelLoaderInvocation = TypeReq<s['MainModelLoaderInvocation']>;
export type OnnxModelLoaderInvocation = TypeReq<s['OnnxModelLoaderInvocation']>;
export type LoraLoaderInvocation = TypeReq<s['LoraLoaderInvocation']>;
export type MetadataAccumulatorInvocation = TypeReq<
  s['MetadataAccumulatorInvocation']
>;
export type ESRGANInvocation = TypeReq<s['ESRGANInvocation']>;
export type DivideInvocation = TypeReq<s['DivideInvocation']>;
export type ImageNSFWBlurInvocation = TypeReq<s['ImageNSFWBlurInvocation']>;
export type ImageWatermarkInvocation = TypeReq<s['ImageWatermarkInvocation']>;

// ControlNet Nodes
export type ControlNetInvocation = TypeReq<s['ControlNetInvocation']>;
export type CannyImageProcessorInvocation = TypeReq<
  s['CannyImageProcessorInvocation']
>;
export type ContentShuffleImageProcessorInvocation = TypeReq<
  s['ContentShuffleImageProcessorInvocation']
>;
export type HedImageProcessorInvocation = TypeReq<
  s['HedImageProcessorInvocation']
>;
export type LineartAnimeImageProcessorInvocation = TypeReq<
  s['LineartAnimeImageProcessorInvocation']
>;
export type LineartImageProcessorInvocation = TypeReq<
  s['LineartImageProcessorInvocation']
>;
export type MediapipeFaceProcessorInvocation = TypeReq<
  s['MediapipeFaceProcessorInvocation']
>;
export type MidasDepthImageProcessorInvocation = TypeReq<
  s['MidasDepthImageProcessorInvocation']
>;
export type MlsdImageProcessorInvocation = TypeReq<
  s['MlsdImageProcessorInvocation']
>;
export type NormalbaeImageProcessorInvocation = TypeReq<
  s['NormalbaeImageProcessorInvocation']
>;
export type OpenposeImageProcessorInvocation = TypeReq<
  s['OpenposeImageProcessorInvocation']
>;
export type PidiImageProcessorInvocation = TypeReq<
  s['PidiImageProcessorInvocation']
>;
export type ZoeDepthImageProcessorInvocation = TypeReq<
  s['ZoeDepthImageProcessorInvocation']
>;

// Node Outputs
export type ImageOutput = s['ImageOutput'];
export type MaskOutput = s['MaskOutput'];
export type PromptOutput = s['PromptOutput'];
export type IterateInvocationOutput = s['IterateInvocationOutput'];
export type CollectInvocationOutput = s['CollectInvocationOutput'];
export type LatentsOutput = s['LatentsOutput'];
export type GraphInvocationOutput = s['GraphInvocationOutput'];

// Post-image upload actions, controls workflows when images are uploaded

export type ControlNetAction = {
  type: 'SET_CONTROLNET_IMAGE';
  controlNetId: string;
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
  | ControlNetAction
  | InitialImageAction
  | NodesAction
  | CanvasInitialImageAction
  | ToastAction
  | AddToBatchAction;

type TypeGuard<T> = {
  (input: unknown): input is T;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type TypeGuardFor<T extends TypeGuard<any>> = T extends TypeGuard<
  infer U
>
  ? U
  : never;
