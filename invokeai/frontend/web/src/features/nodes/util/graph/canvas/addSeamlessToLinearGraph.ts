import type { RootState } from 'app/store/store';
import { upsertMetadata } from 'features/nodes/util/graph/canvas/metadata';
import {
  DENOISE_LATENTS,
  SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH,
  SDXL_CANVAS_INPAINT_GRAPH,
  SDXL_CANVAS_OUTPAINT_GRAPH,
  SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH,
  SDXL_CONTROL_LAYERS_GRAPH,
  SDXL_DENOISE_LATENTS,
  SEAMLESS,
  VAE_LOADER,
} from 'features/nodes/util/graph/constants';
import type { NonNullableGraph } from 'services/api/types';

export const addSeamlessToLinearGraph = (
  state: RootState,
  graph: NonNullableGraph,
  modelLoaderNodeId: string
): void => {
  // Remove Existing UNet Connections
  const { seamlessXAxis, seamlessYAxis, vae } = state.canvasV2.params;
  const isAutoVae = !vae;

  graph.nodes[SEAMLESS] = {
    id: SEAMLESS,
    type: 'seamless',
    seamless_x: seamlessXAxis,
    seamless_y: seamlessYAxis,
  };

  if (!isAutoVae) {
    graph.nodes[VAE_LOADER] = {
      type: 'vae_loader',
      id: VAE_LOADER,
      is_intermediate: true,
      vae_model: vae,
    };
  }

  if (seamlessXAxis) {
    upsertMetadata(graph, {
      seamless_x: seamlessXAxis,
    });
  }
  if (seamlessYAxis) {
    upsertMetadata(graph, {
      seamless_y: seamlessYAxis,
    });
  }

  let denoisingNodeId = DENOISE_LATENTS;

  if (
    graph.id === SDXL_CONTROL_LAYERS_GRAPH ||
    graph.id === SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH ||
    graph.id === SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH ||
    graph.id === SDXL_CANVAS_INPAINT_GRAPH ||
    graph.id === SDXL_CANVAS_OUTPAINT_GRAPH
  ) {
    denoisingNodeId = SDXL_DENOISE_LATENTS;
  }

  graph.edges = graph.edges.filter(
    (e) =>
      !(e.source.node_id === modelLoaderNodeId && ['unet'].includes(e.source.field)) &&
      !(e.source.node_id === modelLoaderNodeId && ['vae'].includes(e.source.field))
  );

  graph.edges.push(
    {
      source: {
        node_id: modelLoaderNodeId,
        field: 'unet',
      },
      destination: {
        node_id: SEAMLESS,
        field: 'unet',
      },
    },
    {
      source: {
        node_id: isAutoVae ? modelLoaderNodeId : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: SEAMLESS,
        field: 'vae',
      },
    },
    {
      source: {
        node_id: SEAMLESS,
        field: 'unet',
      },
      destination: {
        node_id: denoisingNodeId,
        field: 'unet',
      },
    }
  );
};
