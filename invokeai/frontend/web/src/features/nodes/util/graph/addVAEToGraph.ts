import type { RootState } from 'app/store/store';
import type { NonNullableGraph } from 'services/api/types';

import {
  CANVAS_COHERENCE_INPAINT_CREATE_MASK,
  CANVAS_IMAGE_TO_IMAGE_GRAPH,
  CANVAS_INPAINT_GRAPH,
  CANVAS_OUTPAINT_GRAPH,
  CANVAS_OUTPUT,
  CANVAS_TEXT_TO_IMAGE_GRAPH,
  IMAGE_TO_IMAGE_GRAPH,
  IMAGE_TO_LATENTS,
  INPAINT_CREATE_MASK,
  INPAINT_IMAGE,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH,
  SDXL_CANVAS_INPAINT_GRAPH,
  SDXL_CANVAS_OUTPAINT_GRAPH,
  SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH,
  SDXL_IMAGE_TO_IMAGE_GRAPH,
  SDXL_REFINER_INPAINT_CREATE_MASK,
  SDXL_TEXT_TO_IMAGE_GRAPH,
  TEXT_TO_IMAGE_GRAPH,
  VAE_LOADER,
} from './constants';
import { upsertMetadata } from './metadata';

export const addVAEToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  modelLoaderNodeId: string = MAIN_MODEL_LOADER
): void => {
  const { vae, canvasCoherenceMode } = state.generation;
  const { boundingBoxScaleMethod } = state.canvas;
  const { refinerModel } = state.sdxl;

  const isUsingScaledDimensions = ['auto', 'manual'].includes(boundingBoxScaleMethod);

  const isAutoVae = !vae;

  if (!isAutoVae) {
    graph.nodes[VAE_LOADER] = {
      type: 'vae_loader',
      id: VAE_LOADER,
      is_intermediate: true,
      vae_model: vae,
    };
  }

  if (
    graph.id === TEXT_TO_IMAGE_GRAPH ||
    graph.id === IMAGE_TO_IMAGE_GRAPH ||
    graph.id === SDXL_TEXT_TO_IMAGE_GRAPH ||
    graph.id === SDXL_IMAGE_TO_IMAGE_GRAPH
  ) {
    graph.edges.push({
      source: {
        node_id: isAutoVae ? modelLoaderNodeId : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE,
        field: 'vae',
      },
    });
  }

  if (
    graph.id === CANVAS_TEXT_TO_IMAGE_GRAPH ||
    graph.id === CANVAS_IMAGE_TO_IMAGE_GRAPH ||
    graph.id === SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH ||
    graph.id === SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH
  ) {
    graph.edges.push({
      source: {
        node_id: isAutoVae ? modelLoaderNodeId : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: isUsingScaledDimensions ? LATENTS_TO_IMAGE : CANVAS_OUTPUT,
        field: 'vae',
      },
    });
  }

  if (
    graph.id === IMAGE_TO_IMAGE_GRAPH ||
    graph.id === SDXL_IMAGE_TO_IMAGE_GRAPH ||
    graph.id === CANVAS_IMAGE_TO_IMAGE_GRAPH ||
    graph.id === SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH
  ) {
    graph.edges.push({
      source: {
        node_id: isAutoVae ? modelLoaderNodeId : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: IMAGE_TO_LATENTS,
        field: 'vae',
      },
    });
  }

  if (
    graph.id === CANVAS_INPAINT_GRAPH ||
    graph.id === CANVAS_OUTPAINT_GRAPH ||
    graph.id === SDXL_CANVAS_INPAINT_GRAPH ||
    graph.id === SDXL_CANVAS_OUTPAINT_GRAPH
  ) {
    graph.edges.push(
      {
        source: {
          node_id: isAutoVae ? modelLoaderNodeId : VAE_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: INPAINT_IMAGE,
          field: 'vae',
        },
      },
      {
        source: {
          node_id: isAutoVae ? modelLoaderNodeId : VAE_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: INPAINT_CREATE_MASK,
          field: 'vae',
        },
      },
      {
        source: {
          node_id: isAutoVae ? modelLoaderNodeId : VAE_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'vae',
        },
      }
    );

    // Handle Coherence Mode
    if (canvasCoherenceMode !== 'unmasked') {
      graph.edges.push({
        source: {
          node_id: isAutoVae ? modelLoaderNodeId : VAE_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: CANVAS_COHERENCE_INPAINT_CREATE_MASK,
          field: 'vae',
        },
      });
    }
  }

  if (refinerModel) {
    if (graph.id === SDXL_CANVAS_INPAINT_GRAPH || graph.id === SDXL_CANVAS_OUTPAINT_GRAPH) {
      graph.edges.push({
        source: {
          node_id: isAutoVae ? modelLoaderNodeId : VAE_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: SDXL_REFINER_INPAINT_CREATE_MASK,
          field: 'vae',
        },
      });
    }
  }

  if (vae) {
    upsertMetadata(graph, { vae });
  }
};
