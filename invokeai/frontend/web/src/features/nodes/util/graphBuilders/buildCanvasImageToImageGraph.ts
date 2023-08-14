import { logger } from 'app/logging/logger';
import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { initialGenerationState } from 'features/parameters/store/generationSlice';
import {
  ImageDTO,
  ImageResizeInvocation,
  ImageToLatentsInvocation,
} from 'services/api/types';
import { addControlNetToLinearGraph } from './addControlNetToLinearGraph';
import { addDynamicPromptsToGraph } from './addDynamicPromptsToGraph';
import { addLoRAsToGraph } from './addLoRAsToGraph';
import { addNSFWCheckerToGraph } from './addNSFWCheckerToGraph';
import { addVAEToGraph } from './addVAEToGraph';
import { addWatermarkerToGraph } from './addWatermarkerToGraph';
import {
  CANVAS_IMAGE_TO_IMAGE_GRAPH,
  CANVAS_OUTPUT,
  CLIP_SKIP,
  DENOISE_LATENTS,
  IMAGE_TO_LATENTS,
  MAIN_MODEL_LOADER,
  METADATA_ACCUMULATOR,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  RESIZE,
} from './constants';

/**
 * Builds the Canvas tab's Image to Image graph.
 */
export const buildCanvasImageToImageGraph = (
  state: RootState,
  initialImage: ImageDTO
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
    clipSkip,
    shouldUseCpuNoise,
    shouldUseNoiseSettings,
  } = state.generation;

  // The bounding box determines width and height, not the width and height params
  const { width, height } = state.canvas.boundingBoxDimensions;

  const { shouldAutoSave } = state.canvas;

  if (!model) {
    log.error('No model found in state');
    throw new Error('No model found in state');
  }

  const use_cpu = shouldUseNoiseSettings
    ? shouldUseCpuNoise
    : initialGenerationState.shouldUseCpuNoise;

  /**
   * The easiest way to build linear graphs is to do it in the node editor, then copy and paste the
   * full graph here as a template. Then use the parameters from app state and set friendlier node
   * ids.
   *
   * The only thing we need extra logic for is handling randomized seed, control net, and for img2img,
   * the `fit` param. These are added to the graph at the end.
   */

  // copy-pasted graph from node editor, filled in with state values & friendly node ids
  const graph: NonNullableGraph = {
    id: CANVAS_IMAGE_TO_IMAGE_GRAPH,
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
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        is_intermediate: true,
        use_cpu,
      },
      [IMAGE_TO_LATENTS]: {
        type: 'i2l',
        id: IMAGE_TO_LATENTS,
        is_intermediate: true,
        // must be set manually later, bc `fit` parameter may require a resize node inserted
        // image: {
        //   image_name: initialImage.image_name,
        // },
      },
      [DENOISE_LATENTS]: {
        type: 'denoise_latents',
        id: DENOISE_LATENTS,
        is_intermediate: true,
        cfg_scale,
        scheduler,
        steps,
        denoising_start: 1 - strength,
        denoising_end: 1,
      },
      [CANVAS_OUTPUT]: {
        type: 'l2i',
        id: CANVAS_OUTPUT,
        is_intermediate: !shouldAutoSave,
      },
    },
    edges: [
      // Connect Model Loader to CLIP Skip and UNet
      {
        source: {
          node_id: MAIN_MODEL_LOADER,
          field: 'unet',
        },
        destination: {
          node_id: DENOISE_LATENTS,
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
      // Connect CLIP Skip To Conditioning
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
      // Connect Everything To Denoise Latents
      {
        source: {
          node_id: POSITIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'positive_conditioning',
        },
      },
      {
        source: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'negative_conditioning',
        },
      },
      {
        source: {
          node_id: NOISE,
          field: 'noise',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'noise',
        },
      },
      {
        source: {
          node_id: IMAGE_TO_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'latents',
        },
      },
      // Decode the denoised latents to an image
      {
        source: {
          node_id: DENOISE_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'latents',
        },
      },
    ],
  };

  // handle `fit`
  if (initialImage.width !== width || initialImage.height !== height) {
    // The init image needs to be resized to the specified width and height before being passed to `IMAGE_TO_LATENTS`

    // Create a resize node, explicitly setting its image
    const resizeNode: ImageResizeInvocation = {
      id: RESIZE,
      type: 'img_resize',
      image: {
        image_name: initialImage.image_name,
      },
      is_intermediate: true,
      width,
      height,
    };

    graph.nodes[RESIZE] = resizeNode;

    // The `RESIZE` node then passes its image to `IMAGE_TO_LATENTS`
    graph.edges.push({
      source: { node_id: RESIZE, field: 'image' },
      destination: {
        node_id: IMAGE_TO_LATENTS,
        field: 'image',
      },
    });

    // The `RESIZE` node also passes its width and height to `NOISE`
    graph.edges.push({
      source: { node_id: RESIZE, field: 'width' },
      destination: {
        node_id: NOISE,
        field: 'width',
      },
    });

    graph.edges.push({
      source: { node_id: RESIZE, field: 'height' },
      destination: {
        node_id: NOISE,
        field: 'height',
      },
    });
  } else {
    // We are not resizing, so we need to set the image on the `IMAGE_TO_LATENTS` node explicitly
    (graph.nodes[IMAGE_TO_LATENTS] as ImageToLatentsInvocation).image = {
      image_name: initialImage.image_name,
    };

    // Pass the image's dimensions to the `NOISE` node
    graph.edges.push({
      source: { node_id: IMAGE_TO_LATENTS, field: 'width' },
      destination: {
        node_id: NOISE,
        field: 'width',
      },
    });
    graph.edges.push({
      source: { node_id: IMAGE_TO_LATENTS, field: 'height' },
      destination: {
        node_id: NOISE,
        field: 'height',
      },
    });
  }

  // add metadata accumulator, which is only mostly populated - some fields are added later
  graph.nodes[METADATA_ACCUMULATOR] = {
    id: METADATA_ACCUMULATOR,
    type: 'metadata_accumulator',
    generation_mode: 'img2img',
    cfg_scale,
    height,
    width,
    positive_prompt: '', // set in addDynamicPromptsToGraph
    negative_prompt: negativePrompt,
    model,
    seed: 0, // set in addDynamicPromptsToGraph
    steps,
    rand_device: use_cpu ? 'cpu' : 'cuda',
    scheduler,
    vae: undefined, // option; set in addVAEToGraph
    controlnets: [], // populated in addControlNetToLinearGraph
    loras: [], // populated in addLoRAsToGraph
    clip_skip: clipSkip,
    strength,
    init_image: initialImage.image_name,
  };

  graph.edges.push({
    source: {
      node_id: METADATA_ACCUMULATOR,
      field: 'metadata',
    },
    destination: {
      node_id: CANVAS_OUTPUT,
      field: 'metadata',
    },
  });

  // add LoRA support
  addLoRAsToGraph(state, graph, DENOISE_LATENTS);

  // optionally add custom VAE
  addVAEToGraph(state, graph, MAIN_MODEL_LOADER);

  // add dynamic prompts - also sets up core iteration and seed
  addDynamicPromptsToGraph(state, graph);

  // add controlnet, mutating `graph`
  addControlNetToLinearGraph(state, graph, DENOISE_LATENTS);

  // NSFW & watermark - must be last thing added to graph
  if (state.system.shouldUseNSFWChecker) {
    // must add before watermarker!
    addNSFWCheckerToGraph(state, graph, CANVAS_OUTPUT);
  }

  if (state.system.shouldUseWatermarker) {
    // must add after nsfw checker!
    addWatermarkerToGraph(state, graph, CANVAS_OUTPUT);
  }

  return graph;
};
