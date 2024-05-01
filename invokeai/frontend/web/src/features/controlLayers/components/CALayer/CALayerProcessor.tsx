import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { caLayerProcessorConfigChanged, selectCALayer } from 'features/controlLayers/store/controlLayersSlice';
import type { ProcessorConfig } from 'features/controlLayers/util/controlAdapters';
import { memo, useCallback } from 'react';

import { CannyProcessor } from './processors/CannyProcessor';
import { ColorMapProcessor } from './processors/ColorMapProcessor';
import { ContentShuffleProcessor } from './processors/ContentShuffleProcessor';
import { DepthAnythingProcessor } from './processors/DepthAnythingProcessor';
import { DWOpenposeProcessor } from './processors/DWOpenposeProcessor';
import { HedProcessor } from './processors/HedProcessor';
import { LineartProcessor } from './processors/LineartProcessor';
import { MediapipeFaceProcessor } from './processors/MediapipeFaceProcessor';
import { MidasDepthProcessor } from './processors/MidasDepthProcessor';
import { MlsdImageProcessor } from './processors/MlsdImageProcessor';
import { PidiProcessor } from './processors/PidiProcessor';

type Props = {
  layerId: string;
};

export const CALayerProcessor = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const config = useAppSelector((s) => selectCALayer(s.controlLayers.present, layerId).controlAdapter.processorConfig);
  const onChange = useCallback(
    (processorConfig: ProcessorConfig) => {
      dispatch(caLayerProcessorConfigChanged({ layerId, processorConfig }));
    },
    [dispatch, layerId]
  );

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

CALayerProcessor.displayName = 'CALayerProcessor';
