import {
  CannyImageProcessorInvocation,
  ContentShuffleImageProcessorInvocation,
  HedImageprocessorInvocation,
  LineartAnimeImageProcessorInvocation,
  LineartImageProcessorInvocation,
  MediapipeFaceProcessorInvocation,
  MidasDepthImageProcessorInvocation,
  MlsdImageProcessorInvocation,
  NormalbaeImageProcessorInvocation,
  OpenposeImageProcessorInvocation,
  PidiImageProcessorInvocation,
  ZoeDepthImageProcessorInvocation,
} from 'services/api';

export type ControlNetProcessorNode =
  | CannyImageProcessorInvocation
  | HedImageprocessorInvocation
  | LineartImageProcessorInvocation
  | LineartAnimeImageProcessorInvocation
  | OpenposeImageProcessorInvocation
  | MidasDepthImageProcessorInvocation
  | NormalbaeImageProcessorInvocation
  | MlsdImageProcessorInvocation
  | PidiImageProcessorInvocation
  | ContentShuffleImageProcessorInvocation
  | ZoeDepthImageProcessorInvocation
  | MediapipeFaceProcessorInvocation;
