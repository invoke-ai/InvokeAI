import { CannyProcessor } from 'features/controlLayers/components/ControlAndIPAdapter/processors/CannyProcessor';
import { ColorMapProcessor } from 'features/controlLayers/components/ControlAndIPAdapter/processors/ColorMapProcessor';
import { ContentShuffleProcessor } from 'features/controlLayers/components/ControlAndIPAdapter/processors/ContentShuffleProcessor';
import { DepthAnythingProcessor } from 'features/controlLayers/components/ControlAndIPAdapter/processors/DepthAnythingProcessor';
import { DWOpenposeProcessor } from 'features/controlLayers/components/ControlAndIPAdapter/processors/DWOpenposeProcessor';
import { HedProcessor } from 'features/controlLayers/components/ControlAndIPAdapter/processors/HedProcessor';
import { LineartProcessor } from 'features/controlLayers/components/ControlAndIPAdapter/processors/LineartProcessor';
import { MediapipeFaceProcessor } from 'features/controlLayers/components/ControlAndIPAdapter/processors/MediapipeFaceProcessor';
import { MidasDepthProcessor } from 'features/controlLayers/components/ControlAndIPAdapter/processors/MidasDepthProcessor';
import { MlsdImageProcessor } from 'features/controlLayers/components/ControlAndIPAdapter/processors/MlsdImageProcessor';
import { PidiProcessor } from 'features/controlLayers/components/ControlAndIPAdapter/processors/PidiProcessor';
import type { ProcessorConfig } from 'features/controlLayers/store/types';
import { memo } from 'react';

type Props = {
  config: ProcessorConfig | null;
  onChange: (config: ProcessorConfig | null) => void;
};

export const CAProcessorConfig = memo(({ config, onChange }: Props) => {
  if (!config) {
    return null;
  }

  if (config.type === 'canny_image_processor') {
    return <CannyProcessor onChange={onChange} config={config} />;
  }

  if (config.type === 'color_map_image_processor') {
    return <ColorMapProcessor onChange={onChange} config={config} />;
  }

  if (config.type === 'depth_anything_image_processor') {
    return <DepthAnythingProcessor onChange={onChange} config={config} />;
  }

  if (config.type === 'hed_image_processor') {
    return <HedProcessor onChange={onChange} config={config} />;
  }

  if (config.type === 'lineart_image_processor') {
    return <LineartProcessor onChange={onChange} config={config} />;
  }

  if (config.type === 'content_shuffle_image_processor') {
    return <ContentShuffleProcessor onChange={onChange} config={config} />;
  }

  if (config.type === 'lineart_anime_image_processor') {
    // No configurable options for this processor
    return null;
  }

  if (config.type === 'mediapipe_face_processor') {
    return <MediapipeFaceProcessor onChange={onChange} config={config} />;
  }

  if (config.type === 'midas_depth_image_processor') {
    return <MidasDepthProcessor onChange={onChange} config={config} />;
  }

  if (config.type === 'mlsd_image_processor') {
    return <MlsdImageProcessor onChange={onChange} config={config} />;
  }

  if (config.type === 'normalbae_image_processor') {
    // No configurable options for this processor
    return null;
  }

  if (config.type === 'dw_openpose_image_processor') {
    return <DWOpenposeProcessor onChange={onChange} config={config} />;
  }

  if (config.type === 'pidi_image_processor') {
    return <PidiProcessor onChange={onChange} config={config} />;
  }

  if (config.type === 'zoe_depth_image_processor') {
    return null;
  }
});

CAProcessorConfig.displayName = 'CAProcessorConfig';
