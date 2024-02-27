import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterProcessorNode } from 'features/controlAdapters/hooks/useControlAdapterProcessorNode';
import { memo } from 'react';

import CannyProcessor from './processors/CannyProcessor';
import ColorMapProcessor from './processors/ColorMapProcessor';
import ContentShuffleProcessor from './processors/ContentShuffleProcessor';
import DepthAnyThingProcessor from './processors/DepthAnyThingProcessor';
import DWOpenposeProcessor from './processors/DWOpenposeProcessor';
import HedProcessor from './processors/HedProcessor';
import LineartAnimeProcessor from './processors/LineartAnimeProcessor';
import LineartProcessor from './processors/LineartProcessor';
import MediapipeFaceProcessor from './processors/MediapipeFaceProcessor';
import MidasDepthProcessor from './processors/MidasDepthProcessor';
import MlsdImageProcessor from './processors/MlsdImageProcessor';
import NormalBaeProcessor from './processors/NormalBaeProcessor';
import PidiProcessor from './processors/PidiProcessor';
import ZoeDepthProcessor from './processors/ZoeDepthProcessor';

type Props = {
  id: string;
};

const ControlAdapterProcessorComponent = ({ id }: Props) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const processorNode = useControlAdapterProcessorNode(id);

  if (!processorNode) {
    return null;
  }

  if (processorNode.type === 'canny_image_processor') {
    return <CannyProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'color_map_image_processor') {
    return <ColorMapProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'depth_anything_image_processor') {
    return <DepthAnyThingProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'hed_image_processor') {
    return <HedProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'lineart_image_processor') {
    return <LineartProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'content_shuffle_image_processor') {
    return <ContentShuffleProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'lineart_anime_image_processor') {
    return <LineartAnimeProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'mediapipe_face_processor') {
    return <MediapipeFaceProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'midas_depth_image_processor') {
    return <MidasDepthProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'mlsd_image_processor') {
    return <MlsdImageProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'normalbae_image_processor') {
    return <NormalBaeProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'dw_openpose_image_processor') {
    return <DWOpenposeProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'pidi_image_processor') {
    return <PidiProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  if (processorNode.type === 'zoe_depth_image_processor') {
    return <ZoeDepthProcessor controlNetId={id} processorNode={processorNode} isEnabled={isEnabled} />;
  }

  return null;
};

export default memo(ControlAdapterProcessorComponent);
