import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { zImageField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {  selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { selectStartingFrameImage, selectVideoSlice } from 'features/parameters/store/videoSlice';
import { t } from 'i18next';
import { assert } from 'tsafe';

const log = logger('system');

export const buildVeo3VideoGraph = (arg: GraphBuilderArg): GraphBuilderReturn => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Veo3 video graph');

  const supportedModes = ['txt2img'] as const;
  if (!supportedModes.includes(generationMode as any)) {
    throw new UnsupportedGenerationModeError(t('toast.veo3IncompatibleGenerationMode'));
  }

  const params = selectParamsSlice(state);
  const videoParams = selectVideoSlice(state);
  const prompts = selectPresetModifiedPrompts(state);
  assert(prompts.positive.length > 0, 'Veo3 video requires positive prompt to have at least one character');

  
  const { seed, shouldRandomizeSeed } = params;
  const { videoModel } = videoParams;
  const finalSeed = shouldRandomizeSeed ? undefined : seed;

  const g = new Graph(getPrefixedId('veo3_video_graph'));

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
    value: prompts.positive,
  });

  // Create the veo3 video generation node
  const veo3VideoNode = g.addNode({
    id: getPrefixedId('google_veo_3_generate_video'),
    // @ts-expect-error: This node is not available in the OSS application
    type: 'google_veo_3_generate_video',
    model: videoParams.videoModel,
    aspect_ratio: params.dimensions.aspectRatio.id,
    seed: finalSeed,
  });

  const startingFrameImage = selectStartingFrameImage(state);

  if (startingFrameImage) {
    const startingFrameImageField = zImageField.parse(startingFrameImage);
    // @ts-expect-error: This node is not available in the OSS application
    veo3VideoNode.starting_image = startingFrameImageField;
  }

  // @ts-expect-error: This node is not available in the OSS application
  g.addEdge(positivePrompt, 'value', veo3VideoNode, 'prompt');

  // Set up metadata
  g.upsertMetadata({
    positive_prompt: prompts.positive,
    negative_prompt: prompts.negative || '',
    video_duration: params.videoDuration,
    video_aspect_ratio: params.dimensions.aspectRatio.id,
    seed: finalSeed,
    generation_type: 'image-to-video',
    starting_image: startingFrameImage,
    video_model: videoParams.videoModel,
  });

  g.setMetadataReceivingNode(veo3VideoNode);

  return {
    g,
    positivePrompt,
  };
};
