import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { modelIdToVAEModelField } from '../modelIdToVAEModelField';
import {
  IMAGE_TO_IMAGE_GRAPH,
  IMAGE_TO_LATENTS,
  INPAINT,
  INPAINT_GRAPH,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  TEXT_TO_IMAGE_GRAPH,
  VAE_LOADER,
} from './constants';

export const addVAEToGraph = (
  graph: NonNullableGraph,
  state: RootState
): void => {
  const { vae } = state.generation;
  const vae_model = modelIdToVAEModelField(vae?.id || '');

  const isAutoVae = !vae;

  if (!isAutoVae) {
    graph.nodes[VAE_LOADER] = {
      type: 'vae_loader',
      id: VAE_LOADER,
      vae_model,
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
};
