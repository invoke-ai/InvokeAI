import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectVideoFirstFrameImage, selectVideoLastFrameImage } from 'features/parameters/store/videoSlice';
import { zImageField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {  selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { t } from 'i18next';
import { assert } from 'tsafe';

const log = logger('system');

// Default video parameters - these could be moved to a video params slice in the future
const DEFAULT_VIDEO_DURATION = 5;
const DEFAULT_VIDEO_ASPECT_RATIO = "1280:768"; // Default landscape
const DEFAULT_ENHANCE_PROMPT = true;

// Video parameter extraction helper
const getVideoParameters = (state: RootState) => {
  // In the future, these could come from a dedicated video parameters slice
  // For now, we use defaults but allow them to be overridden by any video-specific state
  return {
    duration: DEFAULT_VIDEO_DURATION,
    aspectRatio: DEFAULT_VIDEO_ASPECT_RATIO,
    enhancePrompt: DEFAULT_ENHANCE_PROMPT,
  };
};

export const buildRunwayVideoGraph = (arg: GraphBuilderArg): GraphBuilderReturn => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Runway video graph');

  // Runway video generation supports text-to-video and image-to-video
  // We can support multiple generation modes depending on whether frame images are provided
  const supportedModes = ['txt2img'] as const;
  if (!supportedModes.includes(generationMode as any)) {
    throw new UnsupportedGenerationModeError(t('toast.runwayIncompatibleGenerationMode'));
  }

  const params = selectParamsSlice(state);
  const prompts = selectPresetModifiedPrompts(state);
  const videoFirstFrameImage = selectVideoFirstFrameImage(state);
  const videoLastFrameImage = selectVideoLastFrameImage(state);
  const videoParams = getVideoParameters(state);

  // Get seed from params
  const { seed, shouldRandomizeSeed } = params;
  const finalSeed = shouldRandomizeSeed ? undefined : seed;

  // Determine if this is image-to-video or text-to-video
  const hasFrameImages = videoFirstFrameImage || videoLastFrameImage;

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
    duration: videoParams.duration,
    aspect_ratio: videoParams.aspectRatio,
    seed: finalSeed,
  });

  // @ts-expect-error: This node is not available in the OSS application
  g.addEdge(positivePrompt, 'value', runwayVideoNode, 'prompt');


  // Add first frame image if provided
  if (videoFirstFrameImage) {
    const firstFrameImageField = zImageField.parse(videoFirstFrameImage);
    // @ts-expect-error: This connection is specific to runway node
    runwayVideoNode.first_frame_image = firstFrameImageField;
  }

  // Add last frame image if provided
  if (videoLastFrameImage) {
    const lastFrameImageField = zImageField.parse(videoLastFrameImage);
    // @ts-expect-error: This connection is specific to runway node
    runwayVideoNode.last_frame_image = lastFrameImageField;
  }

  // Set up metadata
  g.upsertMetadata({
    positive_prompt: prompts.positive,
    negative_prompt: prompts.negative || '',
    video_duration: videoParams.duration,
    video_aspect_ratio: videoParams.aspectRatio,
    seed: finalSeed,
    enhance_prompt: videoParams.enhancePrompt,
    generation_type: hasFrameImages ? 'image-to-video' : 'text-to-video',
  });

  // Add video frame images to metadata if they exist
  if (hasFrameImages) {
    g.upsertMetadata({
      first_frame_image: videoFirstFrameImage,
      last_frame_image: videoLastFrameImage,
    }, 'merge');
  }

  g.setMetadataReceivingNode(runwayVideoNode);

  return {
    g,
    positivePrompt,
  };
};
