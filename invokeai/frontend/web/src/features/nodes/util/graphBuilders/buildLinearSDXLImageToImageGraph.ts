import { logger } from 'app/logging/logger';
import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { initialGenerationState } from 'features/parameters/store/generationSlice';
import {
  ImageResizeInvocation,
  ImageToLatentsInvocation,
} from 'services/api/types';
import { addDynamicPromptsToGraph } from './addDynamicPromptsToGraph';
import { addSDXLRefinerToGraph } from './buildSDXLRefinerGraph';
import {
  IMAGE_TO_IMAGE_GRAPH,
  IMAGE_TO_LATENTS,
  LATENTS_TO_IMAGE,
  METADATA_ACCUMULATOR,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  RESIZE,
  SDXL_LATENTS_TO_LATENTS,
  SDXL_MODEL_LOADER,
} from './constants';

/**
 * Builds the Image to Image tab graph.
 */
export const buildLinearSDXLImageToImageGraph = (
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
  } = state.generation;

  const {
    positiveStylePrompt,
    negativeStylePrompt,
    shouldUseSDXLRefiner,
    refinerStart,
  } = state.sdxl;

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

  const use_cpu = shouldUseNoiseSettings
    ? shouldUseCpuNoise
    : initialGenerationState.shouldUseCpuNoise;

  // copy-pasted graph from node editor, filled in with state values & friendly node ids
  const graph: NonNullableGraph = {
    id: IMAGE_TO_IMAGE_GRAPH,
    nodes: {
      [SDXL_MODEL_LOADER]: {
        type: 'sdxl_model_loader',
        id: SDXL_MODEL_LOADER,
        model,
      },
      [POSITIVE_CONDITIONING]: {
        type: 'sdxl_compel_prompt',
        id: POSITIVE_CONDITIONING,
        prompt: positivePrompt,
        style: positiveStylePrompt,
      },
      [NEGATIVE_CONDITIONING]: {
        type: 'sdxl_compel_prompt',
        id: NEGATIVE_CONDITIONING,
        prompt: negativePrompt,
        style: negativeStylePrompt,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        use_cpu,
      },
      [LATENTS_TO_IMAGE]: {
        type: 'l2i',
        id: LATENTS_TO_IMAGE,
      },
      [SDXL_LATENTS_TO_LATENTS]: {
        type: 'l2l_sdxl',
        id: SDXL_LATENTS_TO_LATENTS,
        cfg_scale,
        scheduler,
        steps,
        denoising_start: shouldUseSDXLRefiner ? refinerStart : 1 - strength,
      },
      [IMAGE_TO_LATENTS]: {
        type: 'i2l',
        id: IMAGE_TO_LATENTS,
        // must be set manually later, bc `fit` parameter may require a resize node inserted
        // image: {
        //   image_name: initialImage.image_name,
        // },
      },
    },
    edges: [
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'unet',
        },
        destination: {
          node_id: SDXL_LATENTS_TO_LATENTS,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'vae',
        },
      },
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: IMAGE_TO_LATENTS,
          field: 'vae',
        },
      },
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'clip2',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip2',
        },
      },
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: SDXL_MODEL_LOADER,
          field: 'clip2',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip2',
        },
      },
      {
        source: {
          node_id: SDXL_LATENTS_TO_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'latents',
        },
      },
      {
        source: {
          node_id: IMAGE_TO_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: SDXL_LATENTS_TO_LATENTS,
          field: 'latents',
        },
      },
      {
        source: {
          node_id: NOISE,
          field: 'noise',
        },
        destination: {
          node_id: SDXL_LATENTS_TO_LATENTS,
          field: 'noise',
        },
      },
      {
        source: {
          node_id: POSITIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: SDXL_LATENTS_TO_LATENTS,
          field: 'positive_conditioning',
        },
      },
      {
        source: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: SDXL_LATENTS_TO_LATENTS,
          field: 'negative_conditioning',
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

  // TODO: add batch functionality
  // if (isBatchEnabled && asInitialImage && batchImageNames.length > 0) {
  //   // we are going to connect an iterate up to the init image
  //   delete (graph.nodes[IMAGE_TO_LATENTS] as ImageToLatentsInvocation).image;

  //   const imageCollection: ImageCollectionInvocation = {
  //     id: IMAGE_COLLECTION,
  //     type: 'image_collection',
  //     images: batchImageNames.map((image_name) => ({ image_name })),
  //   };

  //   const imageCollectionIterate: IterateInvocation = {
  //     id: IMAGE_COLLECTION_ITERATE,
  //     type: 'iterate',
  //   };

  //   graph.nodes[IMAGE_COLLECTION] = imageCollection;
  //   graph.nodes[IMAGE_COLLECTION_ITERATE] = imageCollectionIterate;

  //   graph.edges.push({
  //     source: { node_id: IMAGE_COLLECTION, field: 'collection' },
  //     destination: {
  //       node_id: IMAGE_COLLECTION_ITERATE,
  //       field: 'collection',
  //     },
  //   });

  //   graph.edges.push({
  //     source: { node_id: IMAGE_COLLECTION_ITERATE, field: 'item' },
  //     destination: {
  //       node_id: IMAGE_TO_LATENTS,
  //       field: 'image',
  //     },
  //   });
  // }

  // add metadata accumulator, which is only mostly populated - some fields are added later
  graph.nodes[METADATA_ACCUMULATOR] = {
    id: METADATA_ACCUMULATOR,
    type: 'metadata_accumulator',
    generation_mode: 'sdxl_img2img',
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
    vae: undefined,
    controlnets: [],
    loras: [],
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

  // Add Refiner if enabled
  if (shouldUseSDXLRefiner) {
    addSDXLRefinerToGraph(state, graph, SDXL_LATENTS_TO_LATENTS);
  }

  // add dynamic prompts - also sets up core iteration and seed
  addDynamicPromptsToGraph(state, graph);

  return graph;
};
