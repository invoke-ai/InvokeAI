import { logger } from 'app/logging/logger';
import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { initialGenerationState } from 'features/parameters/store/generationSlice';
import {
  ImageResizeInvocation,
  ImageToLatentsInvocation,
} from 'services/api/types';
import { addControlNetToLinearGraph } from './addControlNetToLinearGraph';
import { addDynamicPromptsToGraph } from './addDynamicPromptsToGraph';
import { addLoRAsToGraph } from './addLoRAsToGraph';
import { addNSFWCheckerToGraph } from './addNSFWCheckerToGraph';
import { addSeamlessToLinearGraph } from './addSeamlessToLinearGraph';
import { addVAEToGraph } from './addVAEToGraph';
import { addWatermarkerToGraph } from './addWatermarkerToGraph';
import {
  CLIP_SKIP,
  DENOISE_LATENTS,
  IMAGE_TO_IMAGE_GRAPH,
  IMAGE_TO_LATENTS,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  METADATA_ACCUMULATOR,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  RESIZE,
  SEAMLESS,
} from './constants';

/**
 * Builds the Image to Image tab graph.
 */
export const buildLinearImageToImageGraph = (
  state: RootState
): NonNullableGraph => {
  const log = logger('nodes');
  const {
    positivePrompt,
    negativePrompt,
    model,
    cfgScale: cfg_scale,
    scheduler,
    steps,
    initialImage,
    img2imgStrength: strength,
    shouldFitToWidthHeight,
    width,
    height,
    clipSkip,
    shouldUseCpuNoise,
    shouldUseNoiseSettings,
    vaePrecision,
    seamlessXAxis,
    seamlessYAxis,
  } = state.generation;

  // TODO: add batch functionality
  // const {
  //   isEnabled: isBatchEnabled,
  //   imageNames: batchImageNames,
  //   asInitialImage,
  // } = state.batch;

  // const shouldBatch =
  //   isBatchEnabled && batchImageNames.length > 0 && asInitialImage;

  /**
   * The easiest way to build linear graphs is to do it in the node editor, then copy and paste the
   * full graph here as a template. Then use the parameters from app state and set friendlier node
   * ids.
   *
   * The only thing we need extra logic for is handling randomized seed, control net, and for img2img,
   * the `fit` param. These are added to the graph at the end.
   */

  if (!initialImage) {
    log.error('No initial image found in state');
    throw new Error('No initial image found in state');
  }

  if (!model) {
    log.error('No model found in state');
    throw new Error('No model found in state');
  }

  const fp32 = vaePrecision === 'fp32';

  let modelLoaderNodeId = MAIN_MODEL_LOADER;

  const use_cpu = shouldUseNoiseSettings
    ? shouldUseCpuNoise
    : initialGenerationState.shouldUseCpuNoise;

  // copy-pasted graph from node editor, filled in with state values & friendly node ids
  const graph: NonNullableGraph = {
    id: IMAGE_TO_IMAGE_GRAPH,
    nodes: {
      [modelLoaderNodeId]: {
        type: 'main_model_loader',
        id: modelLoaderNodeId,
        model,
      },
      [CLIP_SKIP]: {
        type: 'clip_skip',
        id: CLIP_SKIP,
        skipped_layers: clipSkip,
      },
      [POSITIVE_CONDITIONING]: {
        type: 'compel',
        id: POSITIVE_CONDITIONING,
        prompt: positivePrompt,
      },
      [NEGATIVE_CONDITIONING]: {
        type: 'compel',
        id: NEGATIVE_CONDITIONING,
        prompt: negativePrompt,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        use_cpu,
      },
      [LATENTS_TO_IMAGE]: {
        type: 'l2i',
        id: LATENTS_TO_IMAGE,
        fp32,
      },
      [DENOISE_LATENTS]: {
        type: 'denoise_latents',
        id: DENOISE_LATENTS,
        cfg_scale,
        scheduler,
        steps,
        denoising_start: 1 - strength,
        denoising_end: 1,
      },
      [IMAGE_TO_LATENTS]: {
        type: 'i2l',
        id: IMAGE_TO_LATENTS,
        // must be set manually later, bc `fit` parameter may require a resize node inserted
        // image: {
        //   image_name: initialImage.image_name,
        // },
        fp32,
      },
    },
    edges: [
      // Connect Model Loader to UNet and CLIP Skip
      {
        source: {
          node_id: modelLoaderNodeId,
          field: 'unet',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: modelLoaderNodeId,
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
      // Connect everything to Denoise Latents
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
      // Decode denoised latents to image
      {
        source: {
          node_id: DENOISE_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'latents',
        },
      },
    ],
  };

  // handle `fit`
  if (
    shouldFitToWidthHeight &&
    (initialImage.width !== width || initialImage.height !== height)
  ) {
    // The init image needs to be resized to the specified width and height before being passed to `IMAGE_TO_LATENTS`

    // Create a resize node, explicitly setting its image
    const resizeNode: ImageResizeInvocation = {
      id: RESIZE,
      type: 'img_resize',
      image: {
        image_name: initialImage.imageName,
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
      image_name: initialImage.imageName,
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
    init_image: initialImage.imageName,
  };

  graph.edges.push({
    source: {
      node_id: METADATA_ACCUMULATOR,
      field: 'metadata',
    },
    destination: {
      node_id: LATENTS_TO_IMAGE,
      field: 'metadata',
    },
  });

  // Add Seamless To Graph
  if (seamlessXAxis || seamlessYAxis) {
    addSeamlessToLinearGraph(state, graph, modelLoaderNodeId);
    modelLoaderNodeId = SEAMLESS;
  }

  // optionally add custom VAE
  addVAEToGraph(state, graph, modelLoaderNodeId);

  // add LoRA support
  addLoRAsToGraph(state, graph, DENOISE_LATENTS, modelLoaderNodeId);

  // add dynamic prompts - also sets up core iteration and seed
  addDynamicPromptsToGraph(state, graph);

  // add controlnet, mutating `graph`
  addControlNetToLinearGraph(state, graph, DENOISE_LATENTS);

  // NSFW & watermark - must be last thing added to graph
  if (state.system.shouldUseNSFWChecker) {
    // must add before watermarker!
    addNSFWCheckerToGraph(state, graph);
  }

  if (state.system.shouldUseWatermarker) {
    // must add after nsfw checker!
    addWatermarkerToGraph(state, graph);
  }

  return graph;
};
