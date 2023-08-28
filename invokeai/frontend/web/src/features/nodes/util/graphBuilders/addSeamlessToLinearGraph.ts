import { RootState } from 'app/store/store';
import { SeamlessModeInvocation } from 'services/api/types';
import { NonNullableGraph } from '../../types/types';
import {
  DENOISE_LATENTS,
  IMAGE_TO_IMAGE_GRAPH,
  SDXL_IMAGE_TO_IMAGE_GRAPH,
  SDXL_TEXT_TO_IMAGE_GRAPH,
  SEAMLESS,
  TEXT_TO_IMAGE_GRAPH,
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

  if (
    graph.id === TEXT_TO_IMAGE_GRAPH ||
    graph.id === IMAGE_TO_IMAGE_GRAPH ||
    graph.id === SDXL_TEXT_TO_IMAGE_GRAPH ||
    graph.id === SDXL_IMAGE_TO_IMAGE_GRAPH
  ) {
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
          node_id: DENOISE_LATENTS,
          field: 'unet',
        },
      }
    );
  }
};
