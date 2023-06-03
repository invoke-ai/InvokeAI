import { RootState } from 'app/store/store';
import {
  CompelInvocation,
  Graph,
  ImageResizeInvocation,
  ImageToLatentsInvocation,
  IterateInvocation,
  LatentsToImageInvocation,
  LatentsToLatentsInvocation,
  NoiseInvocation,
  RandomIntInvocation,
  RangeOfSizeInvocation,
} from 'services/api';
import { NonNullableGraph } from 'features/nodes/types/types';
import { log } from 'app/logging/useLogger';
import { set } from 'lodash-es';

const moduleLog = log.child({ namespace: 'nodes' });

const POSITIVE_CONDITIONING = 'positive_conditioning';
const NEGATIVE_CONDITIONING = 'negative_conditioning';
const IMAGE_TO_LATENTS = 'image_to_latents';
const LATENTS_TO_LATENTS = 'latents_to_latents';
const LATENTS_TO_IMAGE = 'latents_to_image';
const RESIZE = 'resize_image';
const NOISE = 'noise';
const RANDOM_INT = 'rand_int';
const RANGE_OF_SIZE = 'range_of_size';
const ITERATE = 'iterate';

/**
 * Builds the Image to Image tab graph.
 */
export const buildImageToImageGraph = (state: RootState): Graph => {
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
    iterations,
    seed,
    shouldRandomizeSeed,
  } = state.generation;

  if (!initialImage) {
    moduleLog.error('No initial image found in state');
    throw new Error('No initial image found in state');
  }

  const graph: NonNullableGraph = {
    nodes: {},
    edges: [],
  };

  // Create the positive conditioning (prompt) node
  const positiveConditioningNode: CompelInvocation = {
    id: POSITIVE_CONDITIONING,
    type: 'compel',
    prompt: positivePrompt,
    model,
  };

  // Negative conditioning
  const negativeConditioningNode: CompelInvocation = {
    id: NEGATIVE_CONDITIONING,
    type: 'compel',
    prompt: negativePrompt,
    model,
  };

  // This will encode the raster image to latents - but it may get its `image` from a resize node,
  // so we do not set its `image` property yet
  const imageToLatentsNode: ImageToLatentsInvocation = {
    id: IMAGE_TO_LATENTS,
    type: 'i2l',
    model,
  };

  // This does the actual img2img inference
  const latentsToLatentsNode: LatentsToLatentsInvocation = {
    id: LATENTS_TO_LATENTS,
    type: 'l2l',
    cfg_scale,
    model,
    scheduler,
    steps,
    strength,
  };

  // Finally we decode the latents back to an image
  const latentsToImageNode: LatentsToImageInvocation = {
    id: LATENTS_TO_IMAGE,
    type: 'l2i',
    model,
  };

  // Add all those nodes to the graph
  graph.nodes[POSITIVE_CONDITIONING] = positiveConditioningNode;
  graph.nodes[NEGATIVE_CONDITIONING] = negativeConditioningNode;
  graph.nodes[IMAGE_TO_LATENTS] = imageToLatentsNode;
  graph.nodes[LATENTS_TO_LATENTS] = latentsToLatentsNode;
  graph.nodes[LATENTS_TO_IMAGE] = latentsToImageNode;

  // Connect the prompt nodes to the imageToLatents node
  graph.edges.push({
    source: { node_id: POSITIVE_CONDITIONING, field: 'conditioning' },
    destination: {
      node_id: LATENTS_TO_LATENTS,
      field: 'positive_conditioning',
    },
  });
  graph.edges.push({
    source: { node_id: NEGATIVE_CONDITIONING, field: 'conditioning' },
    destination: {
      node_id: LATENTS_TO_LATENTS,
      field: 'negative_conditioning',
    },
  });

  // Connect the image-encoding node
  graph.edges.push({
    source: { node_id: IMAGE_TO_LATENTS, field: 'latents' },
    destination: {
      node_id: LATENTS_TO_LATENTS,
      field: 'latents',
    },
  });

  // Connect the image-decoding node
  graph.edges.push({
    source: { node_id: LATENTS_TO_LATENTS, field: 'latents' },
    destination: {
      node_id: LATENTS_TO_IMAGE,
      field: 'latents',
    },
  });

  /**
   * Now we need to handle iterations and random seeds. There are four possible scenarios:
   * - Single iteration, explicit seed
   * - Single iteration, random seed
   * - Multiple iterations, explicit seed
   * - Multiple iterations, random seed
   *
   * They all have different graphs and connections.
   */

  // Single iteration, explicit seed
  if (!shouldRandomizeSeed && iterations === 1) {
    // Noise node using the explicit seed
    const noiseNode: NoiseInvocation = {
      id: NOISE,
      type: 'noise',
      seed: seed,
    };

    graph.nodes[NOISE] = noiseNode;

    // Connect noise to l2l
    graph.edges.push({
      source: { node_id: NOISE, field: 'noise' },
      destination: {
        node_id: LATENTS_TO_LATENTS,
        field: 'noise',
      },
    });
  }

  // Single iteration, random seed
  if (shouldRandomizeSeed && iterations === 1) {
    // Random int node to generate the seed
    const randomIntNode: RandomIntInvocation = {
      id: RANDOM_INT,
      type: 'rand_int',
    };

    // Noise node without any seed
    const noiseNode: NoiseInvocation = {
      id: NOISE,
      type: 'noise',
    };

    graph.nodes[RANDOM_INT] = randomIntNode;
    graph.nodes[NOISE] = noiseNode;

    // Connect random int to the seed of the noise node
    graph.edges.push({
      source: { node_id: RANDOM_INT, field: 'a' },
      destination: {
        node_id: NOISE,
        field: 'seed',
      },
    });

    // Connect noise to l2l
    graph.edges.push({
      source: { node_id: NOISE, field: 'noise' },
      destination: {
        node_id: LATENTS_TO_LATENTS,
        field: 'noise',
      },
    });
  }

  // Multiple iterations, explicit seed
  if (!shouldRandomizeSeed && iterations > 1) {
    // Range of size node to generate `iterations` count of seeds - range of size generates a collection
    // of ints from `start` to `start + size`. The `start` is the seed, and the `size` is the number of
    // iterations.
    const rangeOfSizeNode: RangeOfSizeInvocation = {
      id: RANGE_OF_SIZE,
      type: 'range_of_size',
      start: seed,
      size: iterations,
    };

    // Iterate node to iterate over the seeds generated by the range of size node
    const iterateNode: IterateInvocation = {
      id: ITERATE,
      type: 'iterate',
    };

    // Noise node without any seed
    const noiseNode: NoiseInvocation = {
      id: NOISE,
      type: 'noise',
    };

    // Adding to the graph
    graph.nodes[RANGE_OF_SIZE] = rangeOfSizeNode;
    graph.nodes[ITERATE] = iterateNode;
    graph.nodes[NOISE] = noiseNode;

    // Connect range of size to iterate
    graph.edges.push({
      source: { node_id: RANGE_OF_SIZE, field: 'collection' },
      destination: {
        node_id: ITERATE,
        field: 'collection',
      },
    });

    // Connect iterate to noise
    graph.edges.push({
      source: {
        node_id: ITERATE,
        field: 'item',
      },
      destination: {
        node_id: NOISE,
        field: 'seed',
      },
    });

    // Connect noise to l2l
    graph.edges.push({
      source: { node_id: NOISE, field: 'noise' },
      destination: {
        node_id: LATENTS_TO_LATENTS,
        field: 'noise',
      },
    });
  }

  // Multiple iterations, random seed
  if (shouldRandomizeSeed && iterations > 1) {
    // Random int node to generate the seed
    const randomIntNode: RandomIntInvocation = {
      id: RANDOM_INT,
      type: 'rand_int',
    };

    // Range of size node to generate `iterations` count of seeds - range of size generates a collection
    const rangeOfSizeNode: RangeOfSizeInvocation = {
      id: RANGE_OF_SIZE,
      type: 'range_of_size',
      size: iterations,
    };

    // Iterate node to iterate over the seeds generated by the range of size node
    const iterateNode: IterateInvocation = {
      id: ITERATE,
      type: 'iterate',
    };

    // Noise node without any seed
    const noiseNode: NoiseInvocation = {
      id: NOISE,
      type: 'noise',
      width,
      height,
    };

    // Adding to the graph
    graph.nodes[RANDOM_INT] = randomIntNode;
    graph.nodes[RANGE_OF_SIZE] = rangeOfSizeNode;
    graph.nodes[ITERATE] = iterateNode;
    graph.nodes[NOISE] = noiseNode;

    // Connect random int to the start of the range of size so the range starts on the random first seed
    graph.edges.push({
      source: { node_id: RANDOM_INT, field: 'a' },
      destination: { node_id: RANGE_OF_SIZE, field: 'start' },
    });

    // Connect range of size to iterate
    graph.edges.push({
      source: { node_id: RANGE_OF_SIZE, field: 'collection' },
      destination: {
        node_id: ITERATE,
        field: 'collection',
      },
    });

    // Connect iterate to noise
    graph.edges.push({
      source: {
        node_id: ITERATE,
        field: 'item',
      },
      destination: {
        node_id: NOISE,
        field: 'seed',
      },
    });

    // Connect noise to l2l
    graph.edges.push({
      source: { node_id: NOISE, field: 'noise' },
      destination: {
        node_id: LATENTS_TO_LATENTS,
        field: 'noise',
      },
    });
  }

  if (shouldFitToWidthHeight) {
    // The init image needs to be resized to the specified width and height before being passed to `IMAGE_TO_LATENTS`

    // Create a resize node, explicitly setting its image
    const resizeNode: ImageResizeInvocation = {
      id: RESIZE,
      type: 'img_resize',
      image: {
        image_name: initialImage.image_name,
        image_origin: initialImage.image_origin,
      },
      is_intermediate: true,
      height,
      width,
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
    set(graph.nodes[IMAGE_TO_LATENTS], 'image', {
      image_name: initialImage.image_name,
      image_origin: initialImage.image_origin,
    });

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

  return graph;
};
