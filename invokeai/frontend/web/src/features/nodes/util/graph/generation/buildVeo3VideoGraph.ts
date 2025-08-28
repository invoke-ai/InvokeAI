import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { zImageField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import {
  selectStartingFrameImage,
  selectVideoModelConfig,
  selectVideoSlice,
} from 'features/parameters/store/videoSlice';
import { t } from 'i18next';
import { assert } from 'tsafe';

const log = logger('system');

export const buildVeo3VideoGraph = (arg: GraphBuilderArg): GraphBuilderReturn => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Veo3 video graph');

  const supportedModes = ['txt2img'];
  if (!supportedModes.includes(generationMode)) {
    throw new UnsupportedGenerationModeError(t('toast.veo3IncompatibleGenerationMode'));
  }

  const model = selectVideoModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'veo3', 'Selected model is not a Veo3 model');

  const params = selectParamsSlice(state);
  const videoParams = selectVideoSlice(state);
  const prompts = selectPresetModifiedPrompts(state);
  assert(prompts.positive.length > 0, 'Veo3 video requires positive prompt to have at least one character');

  const { seed, shouldRandomizeSeed } = params;
  const { videoResolution, videoDuration, videoAspectRatio } = videoParams;
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
    model: model,
    aspect_ratio: '16:9',
    resolution: videoResolution,
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
    model: Graph.getModelMetadataField(model),
    positive_prompt: prompts.positive,
    duration: videoDuration,
    aspect_ratio: videoAspectRatio,
    resolution: videoResolution,
    seed: finalSeed,
    first_frame_image: startingFrameImage,
  });

  g.setMetadataReceivingNode(veo3VideoNode);

  return {
    g,
    positivePrompt,
  };
};
