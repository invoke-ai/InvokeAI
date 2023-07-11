import { O } from 'ts-toolbelt';
import { components } from './schema';

type schemas = components['schemas'];

/**
 * Marks the `type` property as required. Use for nodes.
 */
type TypeReq<T> = O.Required<T, 'type'>;

// App Info
export type AppVersion = components['schemas']['AppVersion'];

// Images
export type ImageDTO = components['schemas']['ImageDTO'];
export type BoardDTO = components['schemas']['BoardDTO'];
export type BoardChanges = components['schemas']['BoardChanges'];
export type ImageChanges = components['schemas']['ImageRecordChanges'];
export type ImageCategory = components['schemas']['ImageCategory'];
export type ResourceOrigin = components['schemas']['ResourceOrigin'];
export type ImageField = components['schemas']['ImageField'];
export type OffsetPaginatedResults_BoardDTO_ =
  components['schemas']['OffsetPaginatedResults_BoardDTO_'];
export type OffsetPaginatedResults_ImageDTO_ =
  components['schemas']['OffsetPaginatedResults_ImageDTO_'];

// Models
export type ModelType = components['schemas']['ModelType'];
export type BaseModelType = components['schemas']['BaseModelType'];
export type MainModelField = components['schemas']['MainModelField'];
export type VAEModelField = components['schemas']['VAEModelField'];
export type LoRAModelField = components['schemas']['LoRAModelField'];
export type ModelsList = components['schemas']['ModelsList'];

// Model Configs
export type LoRAModelConfig = components['schemas']['LoRAModelConfig'];
export type VaeModelConfig = components['schemas']['VaeModelConfig'];
export type ControlNetModelConfig =
  components['schemas']['ControlNetModelConfig'];
export type TextualInversionModelConfig =
  components['schemas']['TextualInversionModelConfig'];
export type MainModelConfig =
  | components['schemas']['StableDiffusion1ModelCheckpointConfig']
  | components['schemas']['StableDiffusion1ModelDiffusersConfig']
  | components['schemas']['StableDiffusion2ModelCheckpointConfig']
  | components['schemas']['StableDiffusion2ModelDiffusersConfig'];
export type AnyModelConfig =
  | LoRAModelConfig
  | VaeModelConfig
  | ControlNetModelConfig
  | TextualInversionModelConfig
  | MainModelConfig;

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
