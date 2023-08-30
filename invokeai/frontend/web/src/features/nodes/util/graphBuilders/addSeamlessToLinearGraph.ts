import { RootState } from 'app/store/store';
import { SeamlessModeInvocation } from 'services/api/types';
import { NonNullableGraph } from '../../types/types';
import {
  CANVAS_COHERENCE_DENOISE_LATENTS,
  CANVAS_INPAINT_GRAPH,
  CANVAS_OUTPAINT_GRAPH,
  DENOISE_LATENTS,
  SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH,
  SDXL_CANVAS_INPAINT_GRAPH,
  SDXL_CANVAS_OUTPAINT_GRAPH,
  SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH,
  SDXL_DENOISE_LATENTS,
  SDXL_IMAGE_TO_IMAGE_GRAPH,
  SDXL_TEXT_TO_IMAGE_GRAPH,
  SEAMLESS,
} from './constants';

export const addSeamlessToLinearGraph = (
  state: RootState,
  graph: NonNullableGraph,
  modelLoaderNodeId: string
): void => {
  // Remove Existing UNet Connections
  const { seamlessXAxis, seamlessYAxis } = state.generation;

  graph.nodes[SEAMLESS] = {
    id: SEAMLESS,
    type: 'seamless',
    seamless_x: seamlessXAxis,
    seamless_y: seamlessYAxis,
  } as SeamlessModeInvocation;

  let denoisingNodeId = DENOISE_LATENTS;

  if (
    graph.id === SDXL_TEXT_TO_IMAGE_GRAPH ||
    graph.id === SDXL_IMAGE_TO_IMAGE_GRAPH ||
    graph.id === SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH ||
    graph.id === SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH ||
    graph.id === SDXL_CANVAS_INPAINT_GRAPH ||
    graph.id === SDXL_CANVAS_OUTPAINT_GRAPH
  ) {
    denoisingNodeId = SDXL_DENOISE_LATENTS;
  }

  graph.edges = graph.edges.filter(
    (e) =>
      !(
        e.source.node_id === modelLoaderNodeId &&
        ['unet'].includes(e.source.field)
      ) &&
      !(
        e.source.node_id === modelLoaderNodeId &&
        ['vae'].includes(e.source.field)
      )
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
        node_id: modelLoaderNodeId,
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

  if (
    graph.id == CANVAS_INPAINT_GRAPH ||
    graph.id === CANVAS_OUTPAINT_GRAPH ||
    graph.id === SDXL_CANVAS_INPAINT_GRAPH ||
    graph.id === SDXL_CANVAS_OUTPAINT_GRAPH
  ) {
    graph.edges.push({
      source: {
        node_id: SEAMLESS,
        field: 'unet',
      },
      destination: {
        node_id: CANVAS_COHERENCE_DENOISE_LATENTS,
        field: 'unet',
      },
    });
  }
};
