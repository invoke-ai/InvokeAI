import { memo } from 'react';
import { ControlNetConfig } from '../store/controlNetSlice';
import CannyProcessor from './processors/CannyProcessor';
import ContentShuffleProcessor from './processors/ContentShuffleProcessor';
import HedProcessor from './processors/HedProcessor';
import LineartAnimeProcessor from './processors/LineartAnimeProcessor';
import LineartProcessor from './processors/LineartProcessor';
import MediapipeFaceProcessor from './processors/MediapipeFaceProcessor';
import MidasDepthProcessor from './processors/MidasDepthProcessor';
import MlsdImageProcessor from './processors/MlsdImageProcessor';
import NormalBaeProcessor from './processors/NormalBaeProcessor';
import OpenposeProcessor from './processors/OpenposeProcessor';
import PidiProcessor from './processors/PidiProcessor';
import ZoeDepthProcessor from './processors/ZoeDepthProcessor';

export type ControlNetProcessorProps = {
  controlNet: ControlNetConfig;
};

const ControlNetProcessorComponent = (props: ControlNetProcessorProps) => {
  const { controlNetId, isEnabled, processorNode } = props.controlNet;

  if (processorNode.type === 'canny_image_processor') {
    return (
      <CannyProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  if (processorNode.type === 'hed_image_processor') {
    return (
      <HedProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  if (processorNode.type === 'lineart_image_processor') {
    return (
      <LineartProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  if (processorNode.type === 'content_shuffle_image_processor') {
    return (
      <ContentShuffleProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  if (processorNode.type === 'lineart_anime_image_processor') {
    return (
      <LineartAnimeProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  if (processorNode.type === 'mediapipe_face_processor') {
    return (
      <MediapipeFaceProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  if (processorNode.type === 'midas_depth_image_processor') {
    return (
      <MidasDepthProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  if (processorNode.type === 'mlsd_image_processor') {
    return (
      <MlsdImageProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  if (processorNode.type === 'normalbae_image_processor') {
    return (
      <NormalBaeProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  if (processorNode.type === 'openpose_image_processor') {
    return (
      <OpenposeProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  if (processorNode.type === 'pidi_image_processor') {
    return (
      <PidiProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  if (processorNode.type === 'zoe_depth_image_processor') {
    return (
      <ZoeDepthProcessor
        controlNetId={controlNetId}
        processorNode={processorNode}
        isEnabled={isEnabled}
      />
    );
  }

  return null;
};

export default memo(ControlNetProcessorComponent);
