/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { AddInvocation } from './AddInvocation';
import type { BlurInvocation } from './BlurInvocation';
import type { CollectInvocation } from './CollectInvocation';
import type { CropImageInvocation } from './CropImageInvocation';
import type { CvInpaintInvocation } from './CvInpaintInvocation';
import type { DivideInvocation } from './DivideInvocation';
import type { Edge } from './Edge';
import type { GraphInvocation } from './GraphInvocation';
import type { ImageToImageInvocation } from './ImageToImageInvocation';
import type { InpaintInvocation } from './InpaintInvocation';
import type { InverseLerpInvocation } from './InverseLerpInvocation';
import type { IterateInvocation } from './IterateInvocation';
import type { LatentsToImageInvocation } from './LatentsToImageInvocation';
import type { LatentsToLatentsInvocation } from './LatentsToLatentsInvocation';
import type { LerpInvocation } from './LerpInvocation';
import type { LoadImageInvocation } from './LoadImageInvocation';
import type { MaskFromAlphaInvocation } from './MaskFromAlphaInvocation';
import type { MultiplyInvocation } from './MultiplyInvocation';
import type { NoiseInvocation } from './NoiseInvocation';
import type { ParamIntInvocation } from './ParamIntInvocation';
import type { PasteImageInvocation } from './PasteImageInvocation';
import type { RandomRangeInvocation } from './RandomRangeInvocation';
import type { RangeInvocation } from './RangeInvocation';
import type { ResizeLatentsInvocation } from './ResizeLatentsInvocation';
import type { RestoreFaceInvocation } from './RestoreFaceInvocation';
import type { ScaleLatentsInvocation } from './ScaleLatentsInvocation';
import type { ShowImageInvocation } from './ShowImageInvocation';
import type { SubtractInvocation } from './SubtractInvocation';
import type { TextToImageInvocation } from './TextToImageInvocation';
import type { TextToLatentsInvocation } from './TextToLatentsInvocation';
import type { UpscaleInvocation } from './UpscaleInvocation';

export type Graph = {
  /**
   * The id of this graph
   */
  id?: string;
  /**
   * The nodes in this graph
   */
  nodes?: Record<string, (LoadImageInvocation | ShowImageInvocation | CropImageInvocation | PasteImageInvocation | MaskFromAlphaInvocation | BlurInvocation | LerpInvocation | InverseLerpInvocation | NoiseInvocation | TextToLatentsInvocation | LatentsToImageInvocation | ResizeLatentsInvocation | ScaleLatentsInvocation | AddInvocation | SubtractInvocation | MultiplyInvocation | DivideInvocation | ParamIntInvocation | CvInpaintInvocation | RangeInvocation | RandomRangeInvocation | UpscaleInvocation | RestoreFaceInvocation | TextToImageInvocation | GraphInvocation | IterateInvocation | CollectInvocation | LatentsToLatentsInvocation | ImageToImageInvocation | InpaintInvocation)>;
  /**
   * The connections between nodes and their fields in this graph
   */
  edges?: Array<Edge>;
};

