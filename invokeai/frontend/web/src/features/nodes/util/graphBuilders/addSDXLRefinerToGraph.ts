import { RootState } from 'app/store/store';
import { MetadataAccumulatorInvocation } from 'services/api/types';
import { NonNullableGraph } from '../../types/types';
import {
  IMAGE_TO_LATENTS,
  LATENTS_TO_IMAGE,
  METADATA_ACCUMULATOR,
  SDXL_LATENTS_TO_LATENTS,
  SDXL_MODEL_LOADER,
  SDXL_REFINER_LATENTS_TO_LATENTS,
  SDXL_REFINER_MODEL_LOADER,
  SDXL_REFINER_NEGATIVE_CONDITIONING,
  SDXL_REFINER_POSITIVE_CONDITIONING,
} from './constants';

export const addSDXLRefinerToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): void => {
  const { positivePrompt, negativePrompt } = state.generation;
  const {
    refinerModel,
    refinerAestheticScore,
    positiveStylePrompt,
    negativeStylePrompt,
    refinerSteps,
    refinerScheduler,
    refinerCFGScale,
    refinerStart,
  } = state.sdxl;

  if (!refinerModel) return;

  const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
    | MetadataAccumulatorInvocation
    | undefined;

  if (metadataAccumulator) {
    metadataAccumulator.refiner_model = refinerModel;
    metadataAccumulator.refiner_aesthetic_store = refinerAestheticScore;
    metadataAccumulator.refiner_cfg_scale = refinerCFGScale;
    metadataAccumulator.refiner_scheduler = refinerScheduler;
    metadataAccumulator.refiner_start = refinerStart;
    metadataAccumulator.refiner_steps = refinerSteps;
  }

  // Unplug SDXL Latents Generation To Latents To Image
  graph.edges = graph.edges.filter(
    (e) =>
      !(e.source.node_id === baseNodeId && ['latents'].includes(e.source.field))
  );

  graph.edges = graph.edges.filter(
    (e) =>
      !(
        e.source.node_id === SDXL_MODEL_LOADER &&
        ['vae'].includes(e.source.field)
      )
  );

  // connect the VAE back to the i2l, which we just removed in the filter
  // but only if we are doing l2l
  if (baseNodeId === SDXL_LATENTS_TO_LATENTS) {
    graph.edges.push({
      source: {
        node_id: SDXL_MODEL_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: IMAGE_TO_LATENTS,
        field: 'vae',
      },
    });
  }

  graph.nodes[SDXL_REFINER_MODEL_LOADER] = {
    type: 'sdxl_refiner_model_loader',
    id: SDXL_REFINER_MODEL_LOADER,
    model: refinerModel,
  };
  graph.nodes[SDXL_REFINER_POSITIVE_CONDITIONING] = {
    type: 'sdxl_refiner_compel_prompt',
    id: SDXL_REFINER_POSITIVE_CONDITIONING,
    style: `${positivePrompt} ${positiveStylePrompt}`,
    aesthetic_score: refinerAestheticScore,
  };
  graph.nodes[SDXL_REFINER_NEGATIVE_CONDITIONING] = {
    type: 'sdxl_refiner_compel_prompt',
    id: SDXL_REFINER_NEGATIVE_CONDITIONING,
    style: `${negativePrompt} ${negativeStylePrompt}`,
    aesthetic_score: refinerAestheticScore,
  };
  graph.nodes[SDXL_REFINER_LATENTS_TO_LATENTS] = {
    type: 'l2l_sdxl',
    id: SDXL_REFINER_LATENTS_TO_LATENTS,
    cfg_scale: refinerCFGScale,
    steps: refinerSteps / (1 - Math.min(refinerStart, 0.99)),
    scheduler: refinerScheduler,
    denoising_start: refinerStart,
    denoising_end: 1,
  };

  graph.edges.push(
    {
      source: {
        node_id: SDXL_REFINER_MODEL_LOADER,
        field: 'unet',
      },
      destination: {
        node_id: SDXL_REFINER_LATENTS_TO_LATENTS,
        field: 'unet',
      },
    },
    {
      source: {
        node_id: SDXL_REFINER_MODEL_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE,
        field: 'vae',
      },
    },
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
        node_id: SDXL_REFINER_LATENTS_TO_LATENTS,
        field: 'positive_conditioning',
      },
    },
    {
      source: {
        node_id: SDXL_REFINER_NEGATIVE_CONDITIONING,
        field: 'conditioning',
      },
      destination: {
        node_id: SDXL_REFINER_LATENTS_TO_LATENTS,
        field: 'negative_conditioning',
      },
    },
    {
      source: {
        node_id: baseNodeId,
        field: 'latents',
      },
      destination: {
        node_id: SDXL_REFINER_LATENTS_TO_LATENTS,
        field: 'latents',
      },
    },
    {
      source: {
        node_id: SDXL_REFINER_LATENTS_TO_LATENTS,
        field: 'latents',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE,
        field: 'latents',
      },
    }
  );
};
