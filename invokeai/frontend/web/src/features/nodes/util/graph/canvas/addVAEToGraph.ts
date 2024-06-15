import type { RootState } from 'app/store/store';
import { upsertMetadata } from 'features/nodes/util/graph/canvas/metadata';
import {
  CANVAS_IMAGE_TO_IMAGE_GRAPH,
  CANVAS_INPAINT_GRAPH,
  CANVAS_OUTPAINT_GRAPH,
  CANVAS_OUTPUT,
  CANVAS_TEXT_TO_IMAGE_GRAPH,
  CONTROL_LAYERS_GRAPH,
  IMAGE_TO_LATENTS,
  INPAINT_CREATE_MASK,
  INPAINT_IMAGE,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH,
  SDXL_CANVAS_INPAINT_GRAPH,
  SDXL_CANVAS_OUTPAINT_GRAPH,
  SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH,
  SDXL_CONTROL_LAYERS_GRAPH,
  SDXL_REFINER_SEAMLESS,
  SEAMLESS,
  VAE_LOADER,
} from 'features/nodes/util/graph/constants';
import type { NonNullableGraph } from 'services/api/types';

export const addVAEToGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  modelLoaderNodeId: string = MAIN_MODEL_LOADER
): Promise<void> => {
  const { vae, seamlessXAxis, seamlessYAxis } = state.canvasV2.params;
  const { boundingBoxScaleMethod } = state.canvas;
  const { refinerModel } = state.canvasV2.params;

  const isUsingScaledDimensions = ['auto', 'manual'].includes(boundingBoxScaleMethod);

  const isAutoVae = !vae;
  const isSeamlessEnabled = seamlessXAxis || seamlessYAxis;
  const isSDXL = Boolean(graph.id?.includes('sdxl'));
  const isUsingRefiner = isSDXL && Boolean(refinerModel);

  if (!isAutoVae && !isSeamlessEnabled) {
    graph.nodes[VAE_LOADER] = {
      type: 'vae_loader',
      id: VAE_LOADER,
      is_intermediate: true,
      vae_model: vae,
    };
  }

  if (graph.id === CONTROL_LAYERS_GRAPH || graph.id === SDXL_CONTROL_LAYERS_GRAPH) {
    graph.edges.push({
      source: {
        node_id: isSeamlessEnabled
          ? isUsingRefiner
            ? SDXL_REFINER_SEAMLESS
            : SEAMLESS
          : isAutoVae
            ? modelLoaderNodeId
            : VAE_LOADER,
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
        node_id: isSeamlessEnabled
          ? isUsingRefiner
            ? SDXL_REFINER_SEAMLESS
            : SEAMLESS
          : isAutoVae
            ? modelLoaderNodeId
            : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: isUsingScaledDimensions ? LATENTS_TO_IMAGE : CANVAS_OUTPUT,
        field: 'vae',
      },
    });
  }

  if (
    (graph.id === CONTROL_LAYERS_GRAPH ||
      graph.id === SDXL_CONTROL_LAYERS_GRAPH ||
      graph.id === CANVAS_IMAGE_TO_IMAGE_GRAPH ||
      graph.id === SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH) &&
    Boolean(graph.nodes[IMAGE_TO_LATENTS])
  ) {
    graph.edges.push({
      source: {
        node_id: isSeamlessEnabled
          ? isUsingRefiner
            ? SDXL_REFINER_SEAMLESS
            : SEAMLESS
          : isAutoVae
            ? modelLoaderNodeId
            : VAE_LOADER,
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
          node_id: isSeamlessEnabled
            ? isUsingRefiner
              ? SDXL_REFINER_SEAMLESS
              : SEAMLESS
            : isAutoVae
              ? modelLoaderNodeId
              : VAE_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: INPAINT_IMAGE,
          field: 'vae',
        },
      },
      {
        source: {
          node_id: isSeamlessEnabled ? SEAMLESS : isAutoVae ? modelLoaderNodeId : VAE_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: INPAINT_CREATE_MASK,
          field: 'vae',
        },
      },

      {
        source: {
          node_id: isSeamlessEnabled
            ? isUsingRefiner
              ? SDXL_REFINER_SEAMLESS
              : SEAMLESS
            : isAutoVae
              ? modelLoaderNodeId
              : VAE_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'vae',
        },
      }
    );
  }

  if (vae) {
    upsertMetadata(graph, { vae });
  }
};
