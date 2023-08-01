import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { MetadataAccumulatorInvocation } from 'services/api/types';
import {
  IMAGE_TO_IMAGE_GRAPH,
  IMAGE_TO_LATENTS,
  INPAINT,
  INPAINT_GRAPH,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  METADATA_ACCUMULATOR,
  TEXT_TO_IMAGE_GRAPH,
  VAE_LOADER,
} from './constants';

export const addVAEToGraph = (
  state: RootState,
  graph: NonNullableGraph
): void => {
  const { vae } = state.generation;

  const isAutoVae = !vae;
  const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
    | MetadataAccumulatorInvocation
    | undefined;

  if (!isAutoVae) {
    graph.nodes[VAE_LOADER] = {
      type: 'vae_loader',
      id: VAE_LOADER,
      is_intermediate: true,
      vae_model: vae,
    };
  }

  if (graph.id === TEXT_TO_IMAGE_GRAPH || graph.id === IMAGE_TO_IMAGE_GRAPH) {
    graph.edges.push({
      source: {
        node_id: isAutoVae ? MAIN_MODEL_LOADER : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE,
        field: 'vae',
      },
    });
  }

  if (graph.id === IMAGE_TO_IMAGE_GRAPH) {
    graph.edges.push({
      source: {
        node_id: isAutoVae ? MAIN_MODEL_LOADER : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: IMAGE_TO_LATENTS,
        field: 'vae',
      },
    });
  }

  if (graph.id === INPAINT_GRAPH) {
    graph.edges.push({
      source: {
        node_id: isAutoVae ? MAIN_MODEL_LOADER : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: INPAINT,
        field: 'vae',
      },
    });
  }

  if (vae && metadataAccumulator) {
    metadataAccumulator.vae = vae;
  }
};
