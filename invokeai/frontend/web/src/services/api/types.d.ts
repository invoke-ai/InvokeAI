import { UseToastOptions } from '@chakra-ui/react';
import { O } from 'ts-toolbelt';
import { components } from './schema';

type schemas = components['schemas'];

/**
 * Marks the `type` property as required. Use for nodes.
 */
type TypeReq<T> = O.Required<T, 'type'>;

// App Info
export type AppVersion = components['schemas']['AppVersion'];
export type AppConfig = components['schemas']['AppConfig'];

// Images
export type ImageDTO = components['schemas']['ImageDTO'];
export type BoardDTO = components['schemas']['BoardDTO'];
export type BoardChanges = components['schemas']['BoardChanges'];
export type ImageChanges = components['schemas']['ImageRecordChanges'];
export type ImageCategory = components['schemas']['ImageCategory'];
export type ResourceOrigin = components['schemas']['ResourceOrigin'];
export type ImageField = components['schemas']['ImageField'];
export type ImageMetadata = components['schemas']['ImageMetadata'];
export type OffsetPaginatedResults_BoardDTO_ =
  components['schemas']['OffsetPaginatedResults_BoardDTO_'];
export type OffsetPaginatedResults_ImageDTO_ =
  components['schemas']['OffsetPaginatedResults_ImageDTO_'];

// Models
export type ModelType = components['schemas']['ModelType'];
export type SubModelType = components['schemas']['SubModelType'];
export type BaseModelType = components['schemas']['BaseModelType'];
export type MainModelField = components['schemas']['MainModelField'];
export type VAEModelField = components['schemas']['VAEModelField'];
export type LoRAModelField = components['schemas']['LoRAModelField'];
export type ControlNetModelField =
  components['schemas']['ControlNetModelField'];
export type ModelsList = components['schemas']['ModelsList'];
export type ControlField = components['schemas']['ControlField'];

// Model Configs
export type LoRAModelConfig = components['schemas']['LoRAModelConfig'];
export type VaeModelConfig = components['schemas']['VaeModelConfig'];
export type ControlNetModelCheckpointConfig =
  components['schemas']['ControlNetModelCheckpointConfig'];
export type ControlNetModelDiffusersConfig =
  components['schemas']['ControlNetModelDiffusersConfig'];
export type ControlNetModelConfig =
  | ControlNetModelCheckpointConfig
  | ControlNetModelDiffusersConfig;
export type TextualInversionModelConfig =
  components['schemas']['TextualInversionModelConfig'];
export type DiffusersModelConfig =
  | components['schemas']['StableDiffusion1ModelDiffusersConfig']
  | components['schemas']['StableDiffusion2ModelDiffusersConfig']
  | components['schemas']['StableDiffusionXLModelDiffusersConfig'];
export type CheckpointModelConfig =
  | components['schemas']['StableDiffusion1ModelCheckpointConfig']
  | components['schemas']['StableDiffusion2ModelCheckpointConfig']
  | components['schemas']['StableDiffusionXLModelCheckpointConfig'];
export type MainModelConfig = DiffusersModelConfig | CheckpointModelConfig;
export type AnyModelConfig =
  | LoRAModelConfig
  | VaeModelConfig
  | ControlNetModelConfig
  | TextualInversionModelConfig
  | MainModelConfig;

export type MergeModelConfig = components['schemas']['Body_merge_models'];
export type ConvertModelConfig = components['schemas']['Body_convert_model'];
export type ImportModelConfig = components['schemas']['Body_import_model'];

// Graphs
export type Graph = components['schemas']['Graph'];
export type Edge = components['schemas']['Edge'];
export type GraphExecutionState = components['schemas']['GraphExecutionState'];

// General nodes
export type CollectInvocation = TypeReq<
  components['schemas']['CollectInvocation']
>;
export type IterateInvocation = TypeReq<
  components['schemas']['IterateInvocation']
>;
export type RangeInvocation = TypeReq<components['schemas']['RangeInvocation']>;
export type RandomRangeInvocation = TypeReq<
  components['schemas']['RandomRangeInvocation']
>;
export type RangeOfSizeInvocation = TypeReq<
  components['schemas']['RangeOfSizeInvocation']
>;
export type InpaintInvocation = TypeReq<
  components['schemas']['InpaintInvocation']
>;
export type ImageResizeInvocation = TypeReq<
  components['schemas']['ImageResizeInvocation']
>;
export type ImageScaleInvocation = TypeReq<
  components['schemas']['ImageScaleInvocation']
>;
export type RandomIntInvocation = TypeReq<
  components['schemas']['RandomIntInvocation']
>;
export type CompelInvocation = TypeReq<
  components['schemas']['CompelInvocation']
>;
export type DynamicPromptInvocation = TypeReq<
  components['schemas']['DynamicPromptInvocation']
>;
export type NoiseInvocation = TypeReq<components['schemas']['NoiseInvocation']>;
export type TextToLatentsInvocation = TypeReq<
  components['schemas']['TextToLatentsInvocation']
>;
export type LatentsToLatentsInvocation = TypeReq<
  components['schemas']['LatentsToLatentsInvocation']
>;
export type ImageToLatentsInvocation = TypeReq<
  components['schemas']['ImageToLatentsInvocation']
>;
export type LatentsToImageInvocation = TypeReq<
  components['schemas']['LatentsToImageInvocation']
>;
export type ImageCollectionInvocation = TypeReq<
  components['schemas']['ImageCollectionInvocation']
>;
export type MainModelLoaderInvocation = TypeReq<
  components['schemas']['MainModelLoaderInvocation']
>;
export type LoraLoaderInvocation = TypeReq<
  components['schemas']['LoraLoaderInvocation']
>;
export type MetadataAccumulatorInvocation = TypeReq<
  components['schemas']['MetadataAccumulatorInvocation']
>;
export type ESRGANInvocation = TypeReq<
  components['schemas']['ESRGANInvocation']
>;
export type DivideInvocation = TypeReq<
  components['schemas']['DivideInvocation']
>;
export type ImageNSFWBlurInvocation = TypeReq<
  components['schemas']['ImageNSFWBlurInvocation']
>;
export type ImageWatermarkInvocation = TypeReq<
  components['schemas']['ImageWatermarkInvocation']
>;

// ControlNet Nodes
export type ControlNetInvocation = TypeReq<
  components['schemas']['ControlNetInvocation']
>;
export type CannyImageProcessorInvocation = TypeReq<
  components['schemas']['CannyImageProcessorInvocation']
>;
export type ContentShuffleImageProcessorInvocation = TypeReq<
  components['schemas']['ContentShuffleImageProcessorInvocation']
>;
export type HedImageProcessorInvocation = TypeReq<
  components['schemas']['HedImageProcessorInvocation']
>;
export type LineartAnimeImageProcessorInvocation = TypeReq<
  components['schemas']['LineartAnimeImageProcessorInvocation']
>;
export type LineartImageProcessorInvocation = TypeReq<
  components['schemas']['LineartImageProcessorInvocation']
>;
export type MediapipeFaceProcessorInvocation = TypeReq<
  components['schemas']['MediapipeFaceProcessorInvocation']
>;
export type MidasDepthImageProcessorInvocation = TypeReq<
  components['schemas']['MidasDepthImageProcessorInvocation']
>;
export type MlsdImageProcessorInvocation = TypeReq<
  components['schemas']['MlsdImageProcessorInvocation']
>;
export type NormalbaeImageProcessorInvocation = TypeReq<
  components['schemas']['NormalbaeImageProcessorInvocation']
>;
export type OpenposeImageProcessorInvocation = TypeReq<
  components['schemas']['OpenposeImageProcessorInvocation']
>;
export type PidiImageProcessorInvocation = TypeReq<
  components['schemas']['PidiImageProcessorInvocation']
>;
export type ZoeDepthImageProcessorInvocation = TypeReq<
  components['schemas']['ZoeDepthImageProcessorInvocation']
>;

// Node Outputs
export type ImageOutput = components['schemas']['ImageOutput'];
export type MaskOutput = components['schemas']['MaskOutput'];
export type PromptOutput = components['schemas']['PromptOutput'];
export type IterateInvocationOutput =
  components['schemas']['IterateInvocationOutput'];
export type CollectInvocationOutput =
  components['schemas']['CollectInvocationOutput'];
export type LatentsOutput = components['schemas']['LatentsOutput'];
export type GraphInvocationOutput =
  components['schemas']['GraphInvocationOutput'];

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
