import { O } from 'ts-toolbelt';
import { components } from './schema';

type schemas = components['schemas'];

/**
 * Extracts the schema type from the schema.
 */
type S<T extends keyof components['schemas']> = components['schemas'][T];

/**
 * Extracts the node type from the schema.
 * Also flags the `type` property as required.
 */
type N<T extends keyof components['schemas']> = O.Required<
  components['schemas'][T],
  'type'
>;

// Images
export type ImageDTO = S<'ImageDTO'>;
export type BoardDTO = S<'BoardDTO'>;
export type BoardChanges = S<'BoardChanges'>;
export type ImageChanges = S<'ImageRecordChanges'>;
export type ImageCategory = S<'ImageCategory'>;
export type ResourceOrigin = S<'ResourceOrigin'>;
export type ImageField = S<'ImageField'>;
export type OffsetPaginatedResults_BoardDTO_ =
  S<'OffsetPaginatedResults_BoardDTO_'>;
export type OffsetPaginatedResults_ImageDTO_ =
  S<'OffsetPaginatedResults_ImageDTO_'>;

// Models
export type ModelType = S<'ModelType'>;
export type BaseModelType = S<'BaseModelType'>;
export type PipelineModelField = S<'PipelineModelField'>;
export type ModelsList = S<'ModelsList'>;

// Graphs
export type Graph = S<'Graph'>;
export type Edge = S<'Edge'>;
export type GraphExecutionState = S<'GraphExecutionState'>;

// General nodes
export type CollectInvocation = N<'CollectInvocation'>;
export type IterateInvocation = N<'IterateInvocation'>;
export type RangeInvocation = N<'RangeInvocation'>;
export type RandomRangeInvocation = N<'RandomRangeInvocation'>;
export type RangeOfSizeInvocation = N<'RangeOfSizeInvocation'>;
export type InpaintInvocation = N<'InpaintInvocation'>;
export type ImageResizeInvocation = N<'ImageResizeInvocation'>;
export type RandomIntInvocation = N<'RandomIntInvocation'>;
export type CompelInvocation = N<'CompelInvocation'>;
export type DynamicPromptInvocation = N<'DynamicPromptInvocation'>;
export type NoiseInvocation = N<'NoiseInvocation'>;
export type TextToLatentsInvocation = N<'TextToLatentsInvocation'>;
export type LatentsToLatentsInvocation = N<'LatentsToLatentsInvocation'>;
export type ImageToLatentsInvocation = N<'ImageToLatentsInvocation'>;
export type LatentsToImageInvocation = N<'LatentsToImageInvocation'>;
export type PipelineModelLoaderInvocation = N<'PipelineModelLoaderInvocation'>;

// ControlNet Nodes
export type ControlNetInvocation = N<'ControlNetInvocation'>;
export type CannyImageProcessorInvocation = N<'CannyImageProcessorInvocation'>;
export type ContentShuffleImageProcessorInvocation =
  N<'ContentShuffleImageProcessorInvocation'>;
export type HedImageProcessorInvocation = N<'HedImageProcessorInvocation'>;
export type LineartAnimeImageProcessorInvocation =
  N<'LineartAnimeImageProcessorInvocation'>;
export type LineartImageProcessorInvocation =
  N<'LineartImageProcessorInvocation'>;
export type MediapipeFaceProcessorInvocation =
  N<'MediapipeFaceProcessorInvocation'>;
export type MidasDepthImageProcessorInvocation =
  N<'MidasDepthImageProcessorInvocation'>;
export type MlsdImageProcessorInvocation = N<'MlsdImageProcessorInvocation'>;
export type NormalbaeImageProcessorInvocation =
  N<'NormalbaeImageProcessorInvocation'>;
export type OpenposeImageProcessorInvocation =
  N<'OpenposeImageProcessorInvocation'>;
export type PidiImageProcessorInvocation = N<'PidiImageProcessorInvocation'>;
export type ZoeDepthImageProcessorInvocation =
  N<'ZoeDepthImageProcessorInvocation'>;

// Node Outputs
export type ImageOutput = S<'ImageOutput'>;
export type MaskOutput = S<'MaskOutput'>;
export type PromptOutput = S<'PromptOutput'>;
export type IterateInvocationOutput = S<'IterateInvocationOutput'>;
export type CollectInvocationOutput = S<'CollectInvocationOutput'>;
export type LatentsOutput = S<'LatentsOutput'>;
export type GraphInvocationOutput = S<'GraphInvocationOutput'>;
