import { logger } from 'app/logging/logger';
import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import {
  ImageDTO,
  InfillPatchmatchInvocation,
  InfillTileInvocation,
  RandomIntInvocation,
  RangeOfSizeInvocation,
} from 'services/api/types';
import { addControlNetToLinearGraph } from './addControlNetToLinearGraph';
import { addLoRAsToGraph } from './addLoRAsToGraph';
import { addNSFWCheckerToGraph } from './addNSFWCheckerToGraph';
import { addVAEToGraph } from './addVAEToGraph';
import { addWatermarkerToGraph } from './addWatermarkerToGraph';
import {
  CLIP_SKIP,
  COLOR_CORRECT,
  INPAINT,
  INPAINT_FINAL_IMAGE,
  INPAINT_GRAPH,
  INPAINT_IMAGE,
  INPAINT_INFILL,
  ITERATE,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  MASK_BLUR,
  MASK_COMBINE,
  MASK_FROM_ALPHA,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  RANDOM_INT,
  RANGE_OF_SIZE,
} from './constants';

/**
 * Builds the Canvas tab's Outpaint graph.
 */
export const buildCanvasOutpaintGraph = (
  state: RootState,
  canvasInitImage: ImageDTO,
  canvasMaskImage?: ImageDTO
): NonNullableGraph => {
  const log = logger('nodes');
  const {
    positivePrompt,
    negativePrompt,
    model,
    cfgScale: cfg_scale,
    scheduler,
    steps,
    img2imgStrength: strength,
    shouldFitToWidthHeight,
    iterations,
    seed,
    shouldRandomizeSeed,
    vaePrecision,
    shouldUseNoiseSettings,
    shouldUseCpuNoise,
    maskBlur,
    maskBlurMethod,
    tileSize,
    infillMethod,
    clipSkip,
  } = state.generation;

  if (!model) {
    log.error('No model found in state');
    throw new Error('No model found in state');
  }

  // The bounding box determines width and height, not the width and height params
  const { width, height } = state.canvas.boundingBoxDimensions;

  // We may need to set the inpaint width and height to scale the image
  const {
    scaledBoundingBoxDimensions,
    boundingBoxScaleMethod,
    shouldAutoSave,
  } = state.canvas;

  const use_cpu = shouldUseNoiseSettings
    ? shouldUseCpuNoise
    : shouldUseCpuNoise;

  let infillNode: InfillTileInvocation | InfillPatchmatchInvocation = {
    type: 'infill_tile',
    id: INPAINT_INFILL,
    is_intermediate: true,
    image: canvasInitImage,
    tile_size: tileSize,
  };

  if (infillMethod === 'patchmatch') {
    infillNode = {
      type: 'infill_patchmatch',
      id: INPAINT_INFILL,
      is_intermediate: true,
      image: canvasInitImage,
    };
  }

  const graph: NonNullableGraph = {
    id: INPAINT_GRAPH,
    nodes: {
      [MAIN_MODEL_LOADER]: {
        type: 'main_model_loader',
        id: MAIN_MODEL_LOADER,
        is_intermediate: true,
        model,
      },
      [CLIP_SKIP]: {
        type: 'clip_skip',
        id: CLIP_SKIP,
        is_intermediate: true,
        skipped_layers: clipSkip,
      },
      [POSITIVE_CONDITIONING]: {
        type: 'compel',
        id: POSITIVE_CONDITIONING,
        is_intermediate: true,
        prompt: positivePrompt,
      },
      [NEGATIVE_CONDITIONING]: {
        type: 'compel',
        id: NEGATIVE_CONDITIONING,
        is_intermediate: true,
        prompt: negativePrompt,
      },
      [MASK_FROM_ALPHA]: {
        type: 'tomask',
        id: MASK_FROM_ALPHA,
        is_intermediate: true,
        image: canvasInitImage,
      },
      [MASK_COMBINE]: {
        type: 'mask_combine',
        id: MASK_COMBINE,
        is_intermediate: true,
        mask2: canvasMaskImage,
      },
      [MASK_BLUR]: {
        type: 'img_blur',
        id: MASK_BLUR,
        is_intermediate: true,
        radius: maskBlur,
        blur_type: maskBlurMethod,
      },
      [infillNode.id]: infillNode,
      [INPAINT_IMAGE]: {
        type: 'i2l',
        id: INPAINT_IMAGE,
        is_intermediate: true,
        fp32: vaePrecision === 'fp32' ? true : false,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        width,
        height,
        use_cpu,
        is_intermediate: true,
      },
      [INPAINT]: {
        type: 'denoise_latents',
        id: INPAINT,
        is_intermediate: true,
        steps: steps,
        cfg_scale: cfg_scale,
        scheduler: scheduler,
        denoising_start: 1 - strength,
        denoising_end: 1,
      },
      [LATENTS_TO_IMAGE]: {
        type: 'l2i',
        id: LATENTS_TO_IMAGE,
        is_intermediate: true,
        fp32: vaePrecision === 'fp32' ? true : false,
      },
      [COLOR_CORRECT]: {
        type: 'color_correct',
        id: COLOR_CORRECT,
        is_intermediate: true,
      },
      [INPAINT_FINAL_IMAGE]: {
        type: 'img_paste',
        id: INPAINT_FINAL_IMAGE,
        is_intermediate: true,
      },
      [RANGE_OF_SIZE]: {
        type: 'range_of_size',
        id: RANGE_OF_SIZE,
        is_intermediate: true,
        // seed - must be connected manually
        // start: 0,
        size: iterations,
        step: 1,
      },
      [ITERATE]: {
        type: 'iterate',
        id: ITERATE,
        is_intermediate: true,
      },
    },
    edges: [
      // Connect Model Loader To UNet & Clip Skip
      {
        source: {
          node_id: MAIN_MODEL_LOADER,
          field: 'unet',
        },
        destination: {
          node_id: INPAINT,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: MAIN_MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: CLIP_SKIP,
          field: 'clip',
        },
      },
      // Connect CLIP Skip to Conditioning
      {
        source: {
          node_id: CLIP_SKIP,
          field: 'clip',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: CLIP_SKIP,
          field: 'clip',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip',
        },
      },
      // Connect Infill Result To Inpaint Image
      {
        source: {
          node_id: INPAINT_INFILL,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_IMAGE,
          field: 'image',
        },
      },
      // Combine Mask from Init Image with User Painted Mask
      {
        source: {
          node_id: MASK_FROM_ALPHA,
          field: 'mask',
        },
        destination: {
          node_id: MASK_COMBINE,
          field: 'mask1',
        },
      },
      {
        source: {
          node_id: MASK_COMBINE,
          field: 'image',
        },
        destination: {
          node_id: MASK_BLUR,
          field: 'image',
        },
      },
      // Plug Everything Into Inpaint Node
      {
        source: {
          node_id: POSITIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: INPAINT,
          field: 'positive_conditioning',
        },
      },
      {
        source: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: INPAINT,
          field: 'negative_conditioning',
        },
      },
      {
        source: {
          node_id: NOISE,
          field: 'noise',
        },
        destination: {
          node_id: INPAINT,
          field: 'noise',
        },
      },
      {
        source: {
          node_id: INPAINT_IMAGE,
          field: 'latents',
        },
        destination: {
          node_id: INPAINT,
          field: 'latents',
        },
      },
      {
        source: {
          node_id: MASK_BLUR,
          field: 'image',
        },
        destination: {
          node_id: INPAINT,
          field: 'mask',
        },
      },
      // Iterate
      {
        source: {
          node_id: RANGE_OF_SIZE,
          field: 'collection',
        },
        destination: {
          node_id: ITERATE,
          field: 'collection',
        },
      },
      {
        source: {
          node_id: ITERATE,
          field: 'item',
        },
        destination: {
          node_id: NOISE,
          field: 'seed',
        },
      },
      // Decode the result from Inpaint
      {
        source: {
          node_id: INPAINT,
          field: 'latents',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'latents',
        },
      },
      // Color Correct The Inpainted Result
      {
        source: {
          node_id: INPAINT_INFILL,
          field: 'image',
        },
        destination: {
          node_id: COLOR_CORRECT,
          field: 'reference',
        },
      },
      {
        source: {
          node_id: LATENTS_TO_IMAGE,
          field: 'image',
        },
        destination: {
          node_id: COLOR_CORRECT,
          field: 'image',
        },
      },
      {
        source: {
          node_id: MASK_BLUR,
          field: 'image',
        },
        destination: {
          node_id: COLOR_CORRECT,
          field: 'mask',
        },
      },
      // Paste Everything Back
      {
        source: {
          node_id: INPAINT_INFILL,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_FINAL_IMAGE,
          field: 'base_image',
        },
      },
      {
        source: {
          node_id: COLOR_CORRECT,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_FINAL_IMAGE,
          field: 'image',
        },
      },
      {
        source: {
          node_id: MASK_BLUR,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_FINAL_IMAGE,
          field: 'mask',
        },
      },
    ],
  };

  // handle seed
  if (shouldRandomizeSeed) {
    // Random int node to generate the starting seed
    const randomIntNode: RandomIntInvocation = {
      id: RANDOM_INT,
      type: 'rand_int',
    };

    graph.nodes[RANDOM_INT] = randomIntNode;

    // Connect random int to the start of the range of size so the range starts on the random first seed
    graph.edges.push({
      source: { node_id: RANDOM_INT, field: 'a' },
      destination: { node_id: RANGE_OF_SIZE, field: 'start' },
    });
  } else {
    // User specified seed, so set the start of the range of size to the seed
    (graph.nodes[RANGE_OF_SIZE] as RangeOfSizeInvocation).start = seed;
  }

  // Add VAE
  addVAEToGraph(state, graph, MAIN_MODEL_LOADER);

  // add LoRA support
  addLoRAsToGraph(state, graph, INPAINT, MAIN_MODEL_LOADER);

  // add controlnet, mutating `graph`
  addControlNetToLinearGraph(state, graph, INPAINT);

  // NSFW & watermark - must be last thing added to graph
  if (state.system.shouldUseNSFWChecker) {
    // must add before watermarker!
    addNSFWCheckerToGraph(state, graph, INPAINT);
  }

  if (state.system.shouldUseWatermarker) {
    // must add after nsfw checker!
    addWatermarkerToGraph(state, graph, INPAINT);
  }

  return graph;
};
