import { memo } from 'react';
import { RequiredControlNetProcessorNode } from '../store/types';
import CannyProcessor from './processors/CannyProcessor';
import HedProcessor from './processors/HedProcessor';
import LineartProcessor from './processors/LineartProcessor';
import LineartAnimeProcessor from './processors/LineartAnimeProcessor';
import ContentShuffleProcessor from './processors/ContentShuffleProcessor';
import MediapipeFaceProcessor from './processors/MediapipeFaceProcessor';
import MidasDepthProcessor from './processors/MidasDepthProcessor';
import MlsdImageProcessor from './processors/MlsdImageProcessor';
import NormalBaeProcessor from './processors/NormalBaeProcessor';
import OpenposeProcessor from './processors/OpenposeProcessor';
import PidiProcessor from './processors/PidiProcessor';
import ZoeDepthProcessor from './processors/ZoeDepthProcessor';

export type ControlNetProcessorProps = {
  controlNetId: string;
  processorNode: RequiredControlNetProcessorNode;
};

const ControlNetProcessorComponent = (props: ControlNetProcessorProps) => {
  const { controlNetId, processorNode } = props;
  if (processorNode.type === 'canny_image_processor') {
    return (
      <CannyProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
      />
    );
  }

  if (processorNode.type === 'hed_image_processor') {
    return (
      <HedProcessor controlNetId={controlNetId} processorNode={processorNode} />
    );
  }

  if (processorNode.type === 'lineart_image_processor') {
    return (
      <LineartProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
      />
    );
  }

  if (processorNode.type === 'content_shuffle_image_processor') {
    return (
      <ContentShuffleProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
      />
    );
  }

  if (processorNode.type === 'lineart_anime_image_processor') {
    return (
      <LineartAnimeProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
      />
    );
  }

  if (processorNode.type === 'mediapipe_face_processor') {
    return (
      <MediapipeFaceProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
      />
    );
  }

  if (processorNode.type === 'midas_depth_image_processor') {
    return (
      <MidasDepthProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
      />
    );
  }

  if (processorNode.type === 'mlsd_image_processor') {
    return (
      <MlsdImageProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
      />
    );
  }

  if (processorNode.type === 'normalbae_image_processor') {
    return (
      <NormalBaeProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
      />
    );
  }

  if (processorNode.type === 'openpose_image_processor') {
    return (
      <OpenposeProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
      />
    );
  }

  if (processorNode.type === 'pidi_image_processor') {
    return (
      <PidiProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
      />
    );
  }

  if (processorNode.type === 'zoe_depth_image_processor') {
    return (
      <ZoeDepthProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
      />
    );
  }

  return null;
};

export default memo(ControlNetProcessorComponent);
