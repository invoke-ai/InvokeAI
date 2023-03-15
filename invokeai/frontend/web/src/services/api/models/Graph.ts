/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BlurInvocation } from './BlurInvocation';
import type { CollectInvocation } from './CollectInvocation';
import type { CropImageInvocation } from './CropImageInvocation';
import type { CvInpaintInvocation } from './CvInpaintInvocation';
import type { Edge } from './Edge';
import type { GraphInvocation } from './GraphInvocation';
import type { ImageToImageInvocation } from './ImageToImageInvocation';
import type { InpaintInvocation } from './InpaintInvocation';
import type { InverseLerpInvocation } from './InverseLerpInvocation';
import type { IterateInvocation } from './IterateInvocation';
import type { LerpInvocation } from './LerpInvocation';
import type { LoadImageInvocation } from './LoadImageInvocation';
import type { MaskFromAlphaInvocation } from './MaskFromAlphaInvocation';
import type { PasteImageInvocation } from './PasteImageInvocation';
import type { RestoreFaceInvocation } from './RestoreFaceInvocation';
import type { ShowImageInvocation } from './ShowImageInvocation';
import type { TextToImageInvocation } from './TextToImageInvocation';
import type { UpscaleInvocation } from './UpscaleInvocation';

export type Graph = {
  /**
   * The id of this graph
   */
  id?: string;
  /**
   * The nodes in this graph
   */
  nodes?: Record<string, (LoadImageInvocation | ShowImageInvocation | CropImageInvocation | PasteImageInvocation | MaskFromAlphaInvocation | BlurInvocation | LerpInvocation | InverseLerpInvocation | CvInpaintInvocation | UpscaleInvocation | RestoreFaceInvocation | TextToImageInvocation | GraphInvocation | IterateInvocation | CollectInvocation | ImageToImageInvocation | InpaintInvocation)>;
  /**
   * The connections between nodes and their fields in this graph
   */
  edges?: Array<Edge>;
};

