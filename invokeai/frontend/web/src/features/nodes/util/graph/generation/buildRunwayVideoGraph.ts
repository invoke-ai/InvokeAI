import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { zImageField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { selectStartingFrameImage, selectVideoSlice } from 'features/parameters/store/videoSlice';
import { t } from 'i18next';
import type { VideoApiModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('system');

export const buildRunwayVideoGraph = (arg: GraphBuilderArg): GraphBuilderReturn => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Runway video graph');

  const supportedModes = ['txt2img'];
  if (!supportedModes.includes(generationMode)) {
    throw new UnsupportedGenerationModeError(t('toast.runwayIncompatibleGenerationMode'));
  }

  const params = selectParamsSlice(state);
  const videoParams = selectVideoSlice(state);
  const prompts = selectPresetModifiedPrompts(state);
  assert(prompts.positive.length > 0, 'Runway video requires positive prompt to have at least one character');

  const startingFrameImage = selectStartingFrameImage(state);

  assert(startingFrameImage, 'Video starting frame is required for runway video generation');
  const firstFrameImageField = zImageField.parse(startingFrameImage);

  const { seed, shouldRandomizeSeed } = params;
  const { videoModel, videoDuration, videoAspectRatio, videoResolution } = videoParams;

  assert(videoModel, 'Runway video requires a model');

  const finalSeed = shouldRandomizeSeed ? undefined : seed;

  const g = new Graph(getPrefixedId('runway_video_graph'));

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
    value: prompts.positive,
  });

  // Create the runway video generation node
  const runwayVideoNode = g.addNode({
    id: getPrefixedId('runway_generate_video'),
    // @ts-expect-error: This node is not available in the OSS application
    type: 'runway_generate_video',
    duration: parseInt(videoDuration || '0', 10),
    aspect_ratio: videoAspectRatio,
    seed: finalSeed,
    first_frame_image: firstFrameImageField,
  });

  // @ts-expect-error: This node is not available in the OSS application
  g.addEdge(positivePrompt, 'value', runwayVideoNode, 'prompt');

  // Set up metadata
  g.upsertMetadata({
    model: Graph.getModelMetadataField(videoModel as VideoApiModelConfig),
    positive_prompt: prompts.positive,
    negative_prompt: prompts.negative || '',
    duration: videoDuration,
    aspect_ratio: videoAspectRatio,
    resolution: videoResolution,
    seed: finalSeed,
    first_frame_image: startingFrameImage,
  });

  g.setMetadataReceivingNode(runwayVideoNode);

  return {
    g,
    positivePrompt,
  };
};
