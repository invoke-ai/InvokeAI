import { log } from 'app/logging/useLogger';
import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { initialGenerationState } from 'features/parameters/store/generationSlice';
import {
  ImageCollectionInvocation,
  ImageResizeInvocation,
  ImageToLatentsInvocation,
  IterateInvocation,
} from 'services/api/types';
import { addControlNetToLinearGraph } from '../addControlNetToLinearGraph';
import { modelIdToMainModelField } from '../modelIdToMainModelField';
import { addDynamicPromptsToGraph } from './addDynamicPromptsToGraph';
import { addLoRAsToGraph } from './addLoRAsToGraph';
import { addVAEToGraph } from './addVAEToGraph';
import {
  CLIP_SKIP,
  IMAGE_COLLECTION,
  IMAGE_COLLECTION_ITERATE,
  IMAGE_TO_IMAGE_GRAPH,
  IMAGE_TO_LATENTS,
  LATENTS_TO_IMAGE,
  LATENTS_TO_LATENTS,
  MAIN_MODEL_LOADER,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  RESIZE,
} from './constants';

const moduleLog = log.child({ namespace: 'nodes' });

/**
 * Builds the Image to Image tab graph.
 */
export const buildLinearImageToImageGraph = (
  state: RootState
): NonNullableGraph => {
  const {
    positivePrompt,
    negativePrompt,
    model: currentModel,
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
    isEnabled: isBatchEnabled,
    imageNames: batchImageNames,
    asInitialImage,
  } = state.batch;

  const shouldBatch =
    isBatchEnabled && batchImageNames.length > 0 && asInitialImage;

  /**
   * The easiest way to build linear graphs is to do it in the node editor, then copy and paste the
   * full graph here as a template. Then use the parameters from app state and set friendlier node
   * ids.
   *
   * The only thing we need extra logic for is handling randomized seed, control net, and for img2img,
   * the `fit` param. These are added to the graph at the end.
   */

  if (!initialImage && !shouldBatch) {
    moduleLog.error('No initial image found in state');
    throw new Error('No initial image found in state');
  }

  const model = modelIdToMainModelField(currentModel?.id || '');

  const use_cpu = shouldUseNoiseSettings
    ? shouldUseCpuNoise
    : initialGenerationState.shouldUseCpuNoise;

  // copy-pasted graph from node editor, filled in with state values & friendly node ids
  const graph: NonNullableGraph = {
    id: IMAGE_TO_IMAGE_GRAPH,
    nodes: {
      [MAIN_MODEL_LOADER]: {
        type: 'main_model_loader',
        id: MAIN_MODEL_LOADER,
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
      },
      [LATENTS_TO_LATENTS]: {
        type: 'l2l',
        id: LATENTS_TO_LATENTS,
        cfg_scale,
        scheduler,
        steps,
        strength,
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
          node_id: MAIN_MODEL_LOADER,
          field: 'unet',
        },
        destination: {
          node_id: LATENTS_TO_LATENTS,
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
      {
        source: {
          node_id: LATENTS_TO_LATENTS,
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
          node_id: LATENTS_TO_LATENTS,
          field: 'latents',
        },
      },
      {
        source: {
          node_id: NOISE,
          field: 'noise',
        },
        destination: {
          node_id: LATENTS_TO_LATENTS,
          field: 'noise',
        },
      },
      {
        source: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: LATENTS_TO_LATENTS,
          field: 'negative_conditioning',
        },
      },
      {
        source: {
          node_id: POSITIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: LATENTS_TO_LATENTS,
          field: 'positive_conditioning',
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

  if (isBatchEnabled && asInitialImage && batchImageNames.length > 0) {
    // we are going to connect an iterate up to the init image
    delete (graph.nodes[IMAGE_TO_LATENTS] as ImageToLatentsInvocation).image;

    const imageCollection: ImageCollectionInvocation = {
      id: IMAGE_COLLECTION,
      type: 'image_collection',
      images: batchImageNames.map((image_name) => ({ image_name })),
    };

    const imageCollectionIterate: IterateInvocation = {
      id: IMAGE_COLLECTION_ITERATE,
      type: 'iterate',
    };

    graph.nodes[IMAGE_COLLECTION] = imageCollection;
    graph.nodes[IMAGE_COLLECTION_ITERATE] = imageCollectionIterate;

    graph.edges.push({
      source: { node_id: IMAGE_COLLECTION, field: 'collection' },
      destination: {
        node_id: IMAGE_COLLECTION_ITERATE,
        field: 'collection',
      },
    });

    graph.edges.push({
      source: { node_id: IMAGE_COLLECTION_ITERATE, field: 'item' },
      destination: {
        node_id: IMAGE_TO_LATENTS,
        field: 'image',
      },
    });
  }

  addLoRAsToGraph(graph, state, LATENTS_TO_LATENTS);

  // Add VAE
  addVAEToGraph(graph, state);

  // add dynamic prompts, mutating `graph`
  addDynamicPromptsToGraph(graph, state);

  // add controlnet, mutating `graph`
  addControlNetToLinearGraph(graph, LATENTS_TO_LATENTS, state);

  return graph;
};
