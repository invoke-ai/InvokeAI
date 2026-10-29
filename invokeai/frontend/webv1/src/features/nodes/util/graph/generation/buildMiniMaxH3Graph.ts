import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import { isMiniMaxH3ReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { zImageField } from 'features/nodes/types/common';
import { addMiniMaxH3LoRAs } from 'features/nodes/util/graph/generation/addMiniMaxH3LoRAs';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getOriginalAndScaledSizesForTextToImage,
  selectCanvasOutputFields,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('system');

/** MiniMax H3 generates at a fixed 24 fps. */
const MINIMAX_H3_FPS = 24;
/** Legal frame counts are 17n+5. 124 frames is the 5s+ video minimum; 345 (14.375s) is the
 * largest grid point within the model's 15s ceiling. The 5-frame minimum block is reserved
 * for the still-image output mode. */
const MINIMAX_H3_MIN_VIDEO_FRAMES = 124;
const MINIMAX_H3_MAX_VIDEO_FRAMES = 345;
const MINIMAX_H3_IMAGE_FRAMES = '5' as const;

/**
 * Snap a duration in seconds to the nearest legal MiniMax H3 frame count (17n+5), clamped to
 * the video range [124, 345].
 */
const snapMiniMaxH3DurationToFrames = (durationSeconds: number): number => {
  // The top slider stop (14 s) maps to the model's true ceiling (345 frames = 14.375 s):
  // nearest-grid rounding alone would top out at 328 and leave the last 0.7 s unreachable.
  if (durationSeconds >= 14) {
    return MINIMAX_H3_MAX_VIDEO_FRAMES;
  }
  const targetFrames = durationSeconds * MINIMAX_H3_FPS;
  const n = Math.round((targetFrames - 5) / 17);
  const frames = n * 17 + 5;
  return Math.min(Math.max(frames, MINIMAX_H3_MIN_VIDEO_FRAMES), MINIMAX_H3_MAX_VIDEO_FRAMES);
};

/**
 * Build a graph for MiniMax H3 (Hailuo 3.0) generation.
 *
 * H3 is a joint audio-video model; the linear UI exposes two output modes:
 * - 'video': text-to-video / first-frame-to-video with a muxed stereo soundtrack. Generate
 *   tab only.
 * - 'image': a minimum-length (5-frame) clip decoded to a single gallery image. This is the
 *   text-to-image mode and also works as the canvas txt2img path.
 *
 * The checkpoint is guidance-distilled: no negative prompt, no CFG - there is exactly one
 * prompt node. When a first-frame reference image is wired, the SAME image and canvas
 * dimensions MUST reach both the text encoder (vision context) and the frame-conditioning
 * node - the backend denoise node enforces this coupling.
 */
export const buildMiniMaxH3Graph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building MiniMax H3 graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'minimax-h3', 'Selected model is not a MiniMax H3 model');
  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'minimax-h3');

  const params = selectParamsSlice(state);
  const { minimaxH3OutputMode } = params;
  // The H3 denoise node requires steps >= 2 (N sigma grid points = N-1 model evaluations);
  // the shared Steps slider allows 1, so clamp rather than 422 at enqueue.
  const steps = Math.max(2, params.steps);

  if (generationMode !== 'txt2img') {
    throw new UnsupportedGenerationModeError(
      'MiniMax H3 supports text-to-video, first-frame-to-video and text-to-image only. ' +
        'Canvas img2img / inpaint / outpaint are not supported.'
    );
  }

  const g = new Graph(getPrefixedId('minimax_h3_graph'));

  // Optional single-file overrides (e.g. pruned int8 transformer, truncated int8 text
  // encoder): each replaces its folder counterpart; everything else still comes from the
  // folder install.
  const transformerModel = params.minimaxH3TransformerModel;
  const textEncoderModel = params.minimaxH3TextEncoderModel;
  const modelLoader = g.addNode({
    type: 'minimax_h3_model_loader',
    id: getPrefixedId('minimax_h3_model_loader'),
    model,
    transformer_model: transformerModel ?? undefined,
    text_encoder_model: textEncoderModel ?? undefined,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  // Guidance-distilled: no negative prompt, no CFG, one text encoder.
  const posCond = g.addNode({
    type: 'minimax_h3_text_encoder',
    id: getPrefixedId('pos_prompt'),
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });

  const denoise = g.addNode({
    type: 'minimax_h3_denoise',
    id: getPrefixedId('denoise_latents'),
    steps,
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'text_encoder', posCond, 'text_encoder');
  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(posCond, 'conditioning', denoise, 'positive_conditioning');
  g.addEdge(seed, 'value', denoise, 'seed');

  // H3 LoRAs (e.g. the Turbo step-distillation LoRA) rewire the transformer edge through a
  // collection loader. Works with both the folder transformer and the single-file overrides.
  addMiniMaxH3LoRAs(state, g, denoise, modelLoader);

  g.upsertMetadata({
    model: Graph.getModelMetadataField(modelConfig),
    steps,
    minimax_h3_output_mode: minimaxH3OutputMode,
  });
  if (transformerModel) {
    g.upsertMetadata({ minimax_h3_transformer_model: transformerModel });
  }
  if (textEncoderModel) {
    g.upsertMetadata({ minimax_h3_text_encoder_model: textEncoderModel });
  }
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  // First-frame conditioning (video mode only): the first enabled MiniMax H3 reference image
  // becomes the video's first frame. The image feeds BOTH the text encoder (vision context)
  // and the frame-conditioning node (VAE condition rows) - the backend requires the pair.
  const refEntity =
    minimaxH3OutputMode === 'video'
      ? selectRefImagesSlice(state).entities.find(
          (entity) =>
            entity.isEnabled &&
            isMiniMaxH3ReferenceImageConfig(entity.config) &&
            entity.config.image !== null &&
            getGlobalReferenceImageWarnings(entity, modelConfig).length === 0
        )
      : undefined;

  if (minimaxH3OutputMode === 'video') {
    if (selectActiveTab(state) !== 'generate') {
      throw new UnsupportedGenerationModeError('MiniMax H3 video generation runs on the Generate tab.');
    }

    const { originalSize } = getOriginalAndScaledSizesForTextToImage(state);
    const num_frames = snapMiniMaxH3DurationToFrames(params.minimaxH3DurationSeconds);

    g.updateNode(denoise, {
      width: originalSize.width,
      height: originalSize.height,
      // The node takes the frame count as a grid-aligned choice value, not a free integer.
      num_frames: `${num_frames}` as Invocation<'minimax_h3_denoise'>['num_frames'],
    });
    // The keyframe vision context is prepared on the same canvas as the condition rows.
    g.updateNode(posCond, {
      width: originalSize.width,
      height: originalSize.height,
    });

    const l2v = g.addNode({
      type: 'minimax_h3_latents_to_video',
      id: getPrefixedId('l2v'),
    });
    g.addEdge(modelLoader, 'vae', l2v, 'vae');
    g.addEdge(modelLoader, 'audio_vae', l2v, 'audio_vae');
    g.addEdge(denoise, 'video_latents', l2v, 'video_latents');
    g.addEdge(denoise, 'audio_latents', l2v, 'audio_latents');

    if (refEntity) {
      assert(isMiniMaxH3ReferenceImageConfig(refEntity.config) && refEntity.config.image);
      const refImageField = zImageField.parse(
        refEntity.config.image.crop?.image ?? refEntity.config.image.original.image
      );
      const frameCond = g.addNode({
        type: 'minimax_h3_frame_conditioning',
        id: getPrefixedId('minimax_h3_frame_conditioning'),
        first_image: refImageField,
        width: originalSize.width,
        height: originalSize.height,
      });
      g.addEdge(modelLoader, 'vae', frameCond, 'vae');
      g.addEdge(frameCond, 'frame_conditioning', denoise, 'frame_conditioning');
      g.updateNode(posCond, { first_image: refImageField });
      g.upsertMetadata({ generation_mode: 'minimax_h3_i2v' });
    } else {
      g.upsertMetadata({ generation_mode: 'minimax_h3_t2v' });
    }

    g.upsertMetadata({
      width: originalSize.width,
      height: originalSize.height,
      minimax_h3_duration_seconds: params.minimaxH3DurationSeconds,
    });

    g.updateNode(l2v, selectCanvasOutputFields(state));
    g.setMetadataReceivingNode(l2v);

    return {
      g,
      seed,
      positivePrompt,
    };
  }

  // Image output mode: a 5-frame (minimum block) clip decoded to a single image. Works on
  // both the Generate tab and canvas txt2img.
  g.updateNode(denoise, { num_frames: MINIMAX_H3_IMAGE_FRAMES });

  const l2i = g.addNode({
    type: 'minimax_h3_latents_to_image',
    id: getPrefixedId('l2i'),
    frame_index: 0,
  });
  g.addEdge(modelLoader, 'vae', l2i, 'vae');
  g.addEdge(denoise, 'video_latents', l2i, 'video_latents');

  let canvasOutput: Invocation<ImageOutputNodes> = addTextToImage({ g, state, denoise, l2i });
  g.upsertMetadata({ generation_mode: 'minimax_h3_txt2img' });

  if (state.system.shouldUseNSFWChecker) {
    canvasOutput = addNSFWChecker(g, canvasOutput);
  }
  if (state.system.shouldUseWatermarker) {
    canvasOutput = addWatermarker(g, canvasOutput);
  }

  g.updateNode(canvasOutput, selectCanvasOutputFields(state));

  if (selectActiveTab(state) === 'canvas') {
    g.upsertMetadata(selectCanvasMetadata(state));
  }

  g.setMetadataReceivingNode(canvasOutput);

  return {
    g,
    seed,
    positivePrompt,
  };
};
