import { RootState } from 'app/store/store';
import {
  MetadataAccumulatorInvocation,
  SeamlessModeInvocation,
} from 'services/api/types';
import { NonNullableGraph } from '../../types/types';
import {
  CANVAS_OUTPUT,
  LATENTS_TO_IMAGE,
  MASK_BLUR,
  METADATA_ACCUMULATOR,
  REFINER_SEAMLESS,
  SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH,
  SDXL_CANVAS_INPAINT_GRAPH,
  SDXL_CANVAS_OUTPAINT_GRAPH,
  SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH,
  SDXL_MODEL_LOADER,
  SDXL_REFINER_DENOISE_LATENTS,
  SDXL_REFINER_MODEL_LOADER,
  SDXL_REFINER_NEGATIVE_CONDITIONING,
  SDXL_REFINER_POSITIVE_CONDITIONING,
} from './constants';
import { craftSDXLStylePrompt } from './helpers/craftSDXLStylePrompt';

export const addSDXLRefinerToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string,
  modelLoaderNodeId?: string
): void => {
  const {
    refinerModel,
    refinerPositiveAestheticScore,
    refinerNegativeAestheticScore,
    refinerSteps,
    refinerScheduler,
    refinerCFGScale,
    refinerStart,
  } = state.sdxl;

  const { seamlessXAxis, seamlessYAxis } = state.generation;

  if (!refinerModel) {
    return;
  }

  const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
    | MetadataAccumulatorInvocation
    | undefined;

  if (metadataAccumulator) {
    metadataAccumulator.refiner_model = refinerModel;
    metadataAccumulator.refiner_positive_aesthetic_store =
      refinerPositiveAestheticScore;
    metadataAccumulator.refiner_negative_aesthetic_store =
      refinerNegativeAestheticScore;
    metadataAccumulator.refiner_cfg_scale = refinerCFGScale;
    metadataAccumulator.refiner_scheduler = refinerScheduler;
    metadataAccumulator.refiner_start = refinerStart;
    metadataAccumulator.refiner_steps = refinerSteps;
  }

  const modelLoaderId = modelLoaderNodeId
    ? modelLoaderNodeId
    : SDXL_MODEL_LOADER;

  // Construct Style Prompt
  const { craftedPositiveStylePrompt, craftedNegativeStylePrompt } =
    craftSDXLStylePrompt(state, true);

  // Unplug SDXL Latents Generation To Latents To Image
  graph.edges = graph.edges.filter(
    (e) =>
      !(e.source.node_id === baseNodeId && ['latents'].includes(e.source.field))
  );

  graph.edges = graph.edges.filter(
    (e) =>
      !(e.source.node_id === modelLoaderId && ['vae'].includes(e.source.field))
  );

  graph.nodes[SDXL_REFINER_MODEL_LOADER] = {
    type: 'sdxl_refiner_model_loader',
    id: SDXL_REFINER_MODEL_LOADER,
    model: refinerModel,
  };
  graph.nodes[SDXL_REFINER_POSITIVE_CONDITIONING] = {
    type: 'sdxl_refiner_compel_prompt',
    id: SDXL_REFINER_POSITIVE_CONDITIONING,
    style: craftedPositiveStylePrompt,
    aesthetic_score: refinerPositiveAestheticScore,
  };
  graph.nodes[SDXL_REFINER_NEGATIVE_CONDITIONING] = {
    type: 'sdxl_refiner_compel_prompt',
    id: SDXL_REFINER_NEGATIVE_CONDITIONING,
    style: craftedNegativeStylePrompt,
    aesthetic_score: refinerNegativeAestheticScore,
  };
  graph.nodes[SDXL_REFINER_DENOISE_LATENTS] = {
    type: 'denoise_latents',
    id: SDXL_REFINER_DENOISE_LATENTS,
    cfg_scale: refinerCFGScale,
    steps: refinerSteps,
    scheduler: refinerScheduler,
    denoising_start: refinerStart,
    denoising_end: 1,
  };

  // Add Seamless To Refiner
  if (seamlessXAxis || seamlessYAxis) {
    graph.nodes[REFINER_SEAMLESS] = {
      id: REFINER_SEAMLESS,
      type: 'seamless',
      seamless_x: seamlessXAxis,
      seamless_y: seamlessYAxis,
    } as SeamlessModeInvocation;

    graph.edges.push(
      {
        source: {
          node_id: SDXL_REFINER_MODEL_LOADER,
          field: 'unet',
        },
        destination: {
          node_id: REFINER_SEAMLESS,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: REFINER_SEAMLESS,
          field: 'unet',
        },
        destination: {
          node_id: SDXL_REFINER_DENOISE_LATENTS,
          field: 'unet',
        },
      }
    );
  } else {
    graph.edges.push({
      source: {
        node_id: SDXL_REFINER_MODEL_LOADER,
        field: 'unet',
      },
      destination: {
        node_id: SDXL_REFINER_DENOISE_LATENTS,
        field: 'unet',
      },
    });
  }

  graph.edges.push(
    {
      source: {
        node_id: SDXL_REFINER_MODEL_LOADER,
        field: 'clip2',
      },
      destination: {
        node_id: SDXL_REFINER_POSITIVE_CONDITIONING,
        field: 'clip2',
      },
    },
    {
      source: {
        node_id: SDXL_REFINER_MODEL_LOADER,
        field: 'clip2',
      },
      destination: {
        node_id: SDXL_REFINER_NEGATIVE_CONDITIONING,
        field: 'clip2',
      },
    },
    {
      source: {
        node_id: SDXL_REFINER_POSITIVE_CONDITIONING,
        field: 'conditioning',
      },
      destination: {
        node_id: SDXL_REFINER_DENOISE_LATENTS,
        field: 'positive_conditioning',
      },
    },
    {
      source: {
        node_id: SDXL_REFINER_NEGATIVE_CONDITIONING,
        field: 'conditioning',
      },
      destination: {
        node_id: SDXL_REFINER_DENOISE_LATENTS,
        field: 'negative_conditioning',
      },
    },
    {
      source: {
        node_id: baseNodeId,
        field: 'latents',
      },
      destination: {
        node_id: SDXL_REFINER_DENOISE_LATENTS,
        field: 'latents',
      },
    }
  );

  if (
    graph.id === SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH ||
    graph.id === SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH
  ) {
    graph.edges.push({
      source: {
        node_id: SDXL_REFINER_DENOISE_LATENTS,
        field: 'latents',
      },
      destination: {
        node_id: CANVAS_OUTPUT,
        field: 'latents',
      },
    });
  } else {
    graph.edges.push({
      source: {
        node_id: SDXL_REFINER_DENOISE_LATENTS,
        field: 'latents',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE,
        field: 'latents',
      },
    });
  }

  if (
    graph.id === SDXL_CANVAS_INPAINT_GRAPH ||
    graph.id === SDXL_CANVAS_OUTPAINT_GRAPH
  ) {
    graph.edges.push({
      source: {
        node_id: MASK_BLUR,
        field: 'image',
      },
      destination: {
        node_id: SDXL_REFINER_DENOISE_LATENTS,
        field: 'mask',
      },
    });
  }
};
