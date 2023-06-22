/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { AddInvocation } from './AddInvocation';
import type { CannyImageProcessorInvocation } from './CannyImageProcessorInvocation';
import type { CollectInvocation } from './CollectInvocation';
import type { CompelInvocation } from './CompelInvocation';
import type { ContentShuffleImageProcessorInvocation } from './ContentShuffleImageProcessorInvocation';
import type { ControlNetInvocation } from './ControlNetInvocation';
import type { CvInpaintInvocation } from './CvInpaintInvocation';
import type { DivideInvocation } from './DivideInvocation';
import type { DynamicPromptInvocation } from './DynamicPromptInvocation';
import type { Edge } from './Edge';
import type { FloatLinearRangeInvocation } from './FloatLinearRangeInvocation';
import type { GraphInvocation } from './GraphInvocation';
import type { HedImageProcessorInvocation } from './HedImageProcessorInvocation';
import type { ImageBlurInvocation } from './ImageBlurInvocation';
import type { ImageChannelInvocation } from './ImageChannelInvocation';
import type { ImageConvertInvocation } from './ImageConvertInvocation';
import type { ImageCropInvocation } from './ImageCropInvocation';
import type { ImageInverseLerpInvocation } from './ImageInverseLerpInvocation';
import type { ImageLerpInvocation } from './ImageLerpInvocation';
import type { ImageMultiplyInvocation } from './ImageMultiplyInvocation';
import type { ImagePasteInvocation } from './ImagePasteInvocation';
import type { ImageProcessorInvocation } from './ImageProcessorInvocation';
import type { ImageResizeInvocation } from './ImageResizeInvocation';
import type { ImageScaleInvocation } from './ImageScaleInvocation';
import type { ImageToLatentsInvocation } from './ImageToLatentsInvocation';
import type { InfillColorInvocation } from './InfillColorInvocation';
import type { InfillPatchMatchInvocation } from './InfillPatchMatchInvocation';
import type { InfillTileInvocation } from './InfillTileInvocation';
import type { InpaintInvocation } from './InpaintInvocation';
import type { IterateInvocation } from './IterateInvocation';
import type { LatentsToImageInvocation } from './LatentsToImageInvocation';
import type { LatentsToLatentsInvocation } from './LatentsToLatentsInvocation';
import type { LineartAnimeImageProcessorInvocation } from './LineartAnimeImageProcessorInvocation';
import type { LineartImageProcessorInvocation } from './LineartImageProcessorInvocation';
import type { LoadImageInvocation } from './LoadImageInvocation';
import type { LoraLoaderInvocation } from './LoraLoaderInvocation';
import type { MaskFromAlphaInvocation } from './MaskFromAlphaInvocation';
import type { MediapipeFaceProcessorInvocation } from './MediapipeFaceProcessorInvocation';
import type { MidasDepthImageProcessorInvocation } from './MidasDepthImageProcessorInvocation';
import type { MlsdImageProcessorInvocation } from './MlsdImageProcessorInvocation';
import type { MultiplyInvocation } from './MultiplyInvocation';
import type { NoiseInvocation } from './NoiseInvocation';
import type { NormalbaeImageProcessorInvocation } from './NormalbaeImageProcessorInvocation';
import type { ONNXLatentsToImageInvocation } from './ONNXLatentsToImageInvocation';
import type { ONNXPromptInvocation } from './ONNXPromptInvocation';
import type { ONNXSD1ModelLoaderInvocation } from './ONNXSD1ModelLoaderInvocation';
import type { ONNXTextToLatentsInvocation } from './ONNXTextToLatentsInvocation';
import type { OpenposeImageProcessorInvocation } from './OpenposeImageProcessorInvocation';
import type { ParamFloatInvocation } from './ParamFloatInvocation';
import type { ParamIntInvocation } from './ParamIntInvocation';
import type { PidiImageProcessorInvocation } from './PidiImageProcessorInvocation';
import type { PipelineModelLoaderInvocation } from './PipelineModelLoaderInvocation';
import type { RandomIntInvocation } from './RandomIntInvocation';
import type { RandomRangeInvocation } from './RandomRangeInvocation';
import type { RangeInvocation } from './RangeInvocation';
import type { RangeOfSizeInvocation } from './RangeOfSizeInvocation';
import type { ResizeLatentsInvocation } from './ResizeLatentsInvocation';
import type { RestoreFaceInvocation } from './RestoreFaceInvocation';
import type { ScaleLatentsInvocation } from './ScaleLatentsInvocation';
import type { ShowImageInvocation } from './ShowImageInvocation';
import type { StepParamEasingInvocation } from './StepParamEasingInvocation';
import type { SubtractInvocation } from './SubtractInvocation';
import type { TextToLatentsInvocation } from './TextToLatentsInvocation';
import type { UpscaleInvocation } from './UpscaleInvocation';
import type { ZoeDepthImageProcessorInvocation } from './ZoeDepthImageProcessorInvocation';

export type Graph = {
  /**
   * The id of this graph
   */
  id?: string;
  /**
   * The nodes in this graph
   */
  nodes?: Record<string, (RangeInvocation | RangeOfSizeInvocation | RandomRangeInvocation | PipelineModelLoaderInvocation | LoraLoaderInvocation | CompelInvocation | LoadImageInvocation | ShowImageInvocation | ImageCropInvocation | ImagePasteInvocation | MaskFromAlphaInvocation | ImageMultiplyInvocation | ImageChannelInvocation | ImageConvertInvocation | ImageBlurInvocation | ImageResizeInvocation | ImageScaleInvocation | ImageLerpInvocation | ImageInverseLerpInvocation | ControlNetInvocation | ImageProcessorInvocation | CvInpaintInvocation | NoiseInvocation | TextToLatentsInvocation | LatentsToImageInvocation | ResizeLatentsInvocation | ScaleLatentsInvocation | ImageToLatentsInvocation | InpaintInvocation | InfillColorInvocation | InfillTileInvocation | InfillPatchMatchInvocation | AddInvocation | SubtractInvocation | MultiplyInvocation | DivideInvocation | RandomIntInvocation | ONNXPromptInvocation | ONNXTextToLatentsInvocation | ONNXLatentsToImageInvocation | ONNXSD1ModelLoaderInvocation | ParamIntInvocation | ParamFloatInvocation | FloatLinearRangeInvocation | StepParamEasingInvocation | DynamicPromptInvocation | RestoreFaceInvocation | UpscaleInvocation | GraphInvocation | IterateInvocation | CollectInvocation | CannyImageProcessorInvocation | HedImageProcessorInvocation | LineartImageProcessorInvocation | LineartAnimeImageProcessorInvocation | OpenposeImageProcessorInvocation | MidasDepthImageProcessorInvocation | NormalbaeImageProcessorInvocation | MlsdImageProcessorInvocation | PidiImageProcessorInvocation | ContentShuffleImageProcessorInvocation | ZoeDepthImageProcessorInvocation | MediapipeFaceProcessorInvocation | LatentsToLatentsInvocation)>;
  /**
   * The connections between nodes and their fields in this graph
   */
  edges?: Array<Edge>;
};
