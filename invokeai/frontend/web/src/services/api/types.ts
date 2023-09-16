import { UseToastOptions } from '@chakra-ui/react';
import { EntityState } from '@reduxjs/toolkit';
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
export type ONNXTextToLatentsInvocation = s['ONNXTextToLatentsInvocation'];
export type SDXLLoraLoaderInvocation = s['SDXLLoraLoaderInvocation'];
export type ImageToLatentsInvocation = s['ImageToLatentsInvocation'];
export type LatentsToImageInvocation = s['LatentsToImageInvocation'];
export type ImageCollectionInvocation = s['ImageCollectionInvocation'];
export type MainModelLoaderInvocation = s['MainModelLoaderInvocation'];
export type OnnxModelLoaderInvocation = s['OnnxModelLoaderInvocation'];
export type LoraLoaderInvocation = s['LoraLoaderInvocation'];
export type MetadataAccumulatorInvocation = s['MetadataAccumulatorInvocation'];
export type ESRGANInvocation = s['ESRGANInvocation'];
export type DivideInvocation = s['DivideInvocation'];
export type ImageNSFWBlurInvocation = s['ImageNSFWBlurInvocation'];
export type ImageWatermarkInvocation = s['ImageWatermarkInvocation'];
export type SeamlessModeInvocation = s['SeamlessModeInvocation'];

// ControlNet Nodes
export type ControlNetInvocation = s['ControlNetInvocation'];
export type CannyImageProcessorInvocation = s['CannyImageProcessorInvocation'];
export type ContentShuffleImageProcessorInvocation =
  s['ContentShuffleImageProcessorInvocation'];
export type HedImageProcessorInvocation = s['HedImageProcessorInvocation'];
export type LineartAnimeImageProcessorInvocation =
  s['LineartAnimeImageProcessorInvocation'];
export type LineartImageProcessorInvocation =
  s['LineartImageProcessorInvocation'];
export type MediapipeFaceProcessorInvocation =
  s['MediapipeFaceProcessorInvocation'];
export type MidasDepthImageProcessorInvocation =
  s['MidasDepthImageProcessorInvocation'];
export type MlsdImageProcessorInvocation = s['MlsdImageProcessorInvocation'];
export type NormalbaeImageProcessorInvocation =
  s['NormalbaeImageProcessorInvocation'];
export type OpenposeImageProcessorInvocation =
  s['OpenposeImageProcessorInvocation'];
export type PidiImageProcessorInvocation = s['PidiImageProcessorInvocation'];
export type ZoeDepthImageProcessorInvocation =
  s['ZoeDepthImageProcessorInvocation'];

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
