import { RootState } from 'app/store/store';
import {
  ImageDTO,
  InpaintInvocation,
  RandomIntInvocation,
  RangeOfSizeInvocation,
} from 'services/api/types';
import { NonNullableGraph } from 'features/nodes/types/types';
import { log } from 'app/logging/useLogger';
import {
  ITERATE,
  MODEL_LOADER,
  NEGATIVE_CONDITIONING,
  POSITIVE_CONDITIONING,
  RANDOM_INT,
  RANGE_OF_SIZE,
  INPAINT_GRAPH,
  INPAINT,
} from './constants';
import { modelIdToPipelineModelField } from '../modelIdToPipelineModelField';

const moduleLog = log.child({ namespace: 'nodes' });

/**
 * Builds the Canvas tab's Inpaint graph.
 */
export const buildCanvasInpaintGraph = (
  state: RootState,
  canvasInitImage: ImageDTO,
  canvasMaskImage: ImageDTO
): NonNullableGraph => {
  const {
    positivePrompt,
    negativePrompt,
    model: modelId,
    cfgScale: cfg_scale,
    scheduler,
    steps,
    img2imgStrength: strength,
    shouldFitToWidthHeight,
    iterations,
    seed,
    shouldRandomizeSeed,
    seamSize,
    seamBlur,
    seamSteps,
    seamStrength,
    tileSize,
    infillMethod,
  } = state.generation;

  // The bounding box determines width and height, not the width and height params
  const { width, height } = state.canvas.boundingBoxDimensions;

  // We may need to set the inpaint width and height to scale the image
  const { scaledBoundingBoxDimensions, boundingBoxScaleMethod } = state.canvas;

  const model = modelIdToPipelineModelField(modelId);

  const graph: NonNullableGraph = {
    id: INPAINT_GRAPH,
    nodes: {
      [INPAINT]: {
        type: 'inpaint',
        id: INPAINT,
        steps,
        width,
        height,
        cfg_scale,
        scheduler,
        image: {
          image_name: canvasInitImage.image_name,
        },
        strength,
        fit: shouldFitToWidthHeight,
        mask: {
          image_name: canvasMaskImage.image_name,
        },
        seam_size: seamSize,
        seam_blur: seamBlur,
        seam_strength: seamStrength,
        seam_steps: seamSteps,
        tile_size: infillMethod === 'tile' ? tileSize : undefined,
        infill_method: infillMethod as InpaintInvocation['infill_method'],
        inpaint_width:
          boundingBoxScaleMethod !== 'none'
            ? scaledBoundingBoxDimensions.width
            : undefined,
        inpaint_height:
          boundingBoxScaleMethod !== 'none'
            ? scaledBoundingBoxDimensions.height
            : undefined,
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
      [MODEL_LOADER]: {
        type: 'pipeline_model_loader',
        id: MODEL_LOADER,
        model,
      },
      [RANGE_OF_SIZE]: {
        type: 'range_of_size',
        id: RANGE_OF_SIZE,
        // seed - must be connected manually
        // start: 0,
        size: iterations,
        step: 1,
      },
      [ITERATE]: {
        type: 'iterate',
        id: ITERATE,
      },
    },
    edges: [
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
          node_id: MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'unet',
        },
        destination: {
          node_id: INPAINT,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: INPAINT,
          field: 'vae',
        },
      },
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
          node_id: INPAINT,
          field: 'seed',
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

  return graph;
};
