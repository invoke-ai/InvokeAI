import type { RootState } from 'app/store/store';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import {
  type CreateDenoiseMaskInvocation,
  type ImageDTO,
  isRefinerMainModelModelConfig,
  type NonNullableGraph,
  type SeamlessModeInvocation,
} from 'services/api/types';

import {
  CANVAS_OUTPUT,
  INPAINT_IMAGE_RESIZE_UP,
  LATENTS_TO_IMAGE,
  MASK_COMBINE,
  MASK_RESIZE_UP,
  SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH,
  SDXL_CANVAS_INPAINT_GRAPH,
  SDXL_CANVAS_OUTPAINT_GRAPH,
  SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH,
  SDXL_MODEL_LOADER,
  SDXL_REFINER_DENOISE_LATENTS,
  SDXL_REFINER_INPAINT_CREATE_MASK,
  SDXL_REFINER_MODEL_LOADER,
  SDXL_REFINER_NEGATIVE_CONDITIONING,
  SDXL_REFINER_POSITIVE_CONDITIONING,
  SDXL_REFINER_SEAMLESS,
} from './constants';
import { getSDXLStylePrompts } from './graphBuilderUtils';
import { getModelMetadataField, upsertMetadata } from './metadata';

export const addSDXLRefinerToGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string,
  modelLoaderNodeId?: string,
  canvasInitImage?: ImageDTO,
  canvasMaskImage?: ImageDTO
): Promise<void> => {
  const {
    refinerModel,
    refinerPositiveAestheticScore,
    refinerNegativeAestheticScore,
    refinerSteps,
    refinerScheduler,
    refinerCFGScale,
    refinerStart,
  } = state.sdxl;

  if (!refinerModel) {
    return;
  }

  const { seamlessXAxis, seamlessYAxis, vaePrecision } = state.generation;
  const { boundingBoxScaleMethod } = state.canvas;

  const fp32 = vaePrecision === 'fp32';

  const isUsingScaledDimensions = ['auto', 'manual'].includes(boundingBoxScaleMethod);
  const modelConfig = await fetchModelConfigWithTypeGuard(refinerModel.key, isRefinerMainModelModelConfig);

  upsertMetadata(graph, {
    refiner_model: getModelMetadataField(modelConfig),
    refiner_positive_aesthetic_score: refinerPositiveAestheticScore,
    refiner_negative_aesthetic_score: refinerNegativeAestheticScore,
    refiner_cfg_scale: refinerCFGScale,
    refiner_scheduler: refinerScheduler,
    refiner_start: refinerStart,
    refiner_steps: refinerSteps,
  });

  const modelLoaderId = modelLoaderNodeId ? modelLoaderNodeId : SDXL_MODEL_LOADER;

  // Construct Style Prompt
  const { positiveStylePrompt, negativeStylePrompt } = getSDXLStylePrompts(state);

  // Unplug SDXL Latents Generation To Latents To Image
  graph.edges = graph.edges.filter((e) => !(e.source.node_id === baseNodeId && ['latents'].includes(e.source.field)));

  graph.edges = graph.edges.filter((e) => !(e.source.node_id === modelLoaderId && ['vae'].includes(e.source.field)));

  graph.nodes[SDXL_REFINER_MODEL_LOADER] = {
    type: 'sdxl_refiner_model_loader',
    id: SDXL_REFINER_MODEL_LOADER,
    model: refinerModel,
  };
  graph.nodes[SDXL_REFINER_POSITIVE_CONDITIONING] = {
    type: 'sdxl_refiner_compel_prompt',
    id: SDXL_REFINER_POSITIVE_CONDITIONING,
    style: positiveStylePrompt,
    aesthetic_score: refinerPositiveAestheticScore,
  };
  graph.nodes[SDXL_REFINER_NEGATIVE_CONDITIONING] = {
    type: 'sdxl_refiner_compel_prompt',
    id: SDXL_REFINER_NEGATIVE_CONDITIONING,
    style: negativeStylePrompt,
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
    graph.nodes[SDXL_REFINER_SEAMLESS] = {
      id: SDXL_REFINER_SEAMLESS,
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
          node_id: SDXL_REFINER_SEAMLESS,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: SDXL_REFINER_MODEL_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: SDXL_REFINER_SEAMLESS,
          field: 'vae',
        },
      },
      {
        source: {
          node_id: SDXL_REFINER_SEAMLESS,
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

  if (graph.id === SDXL_CANVAS_INPAINT_GRAPH || graph.id === SDXL_CANVAS_OUTPAINT_GRAPH) {
    graph.nodes[SDXL_REFINER_INPAINT_CREATE_MASK] = {
      type: 'create_denoise_mask',
      id: SDXL_REFINER_INPAINT_CREATE_MASK,
      is_intermediate: true,
      fp32,
    };

    if (isUsingScaledDimensions) {
      graph.edges.push({
        source: {
          node_id: INPAINT_IMAGE_RESIZE_UP,
          field: 'image',
        },
        destination: {
          node_id: SDXL_REFINER_INPAINT_CREATE_MASK,
          field: 'image',
        },
      });
    } else {
      graph.nodes[SDXL_REFINER_INPAINT_CREATE_MASK] = {
        ...(graph.nodes[SDXL_REFINER_INPAINT_CREATE_MASK] as CreateDenoiseMaskInvocation),
        image: canvasInitImage,
      };
    }

    if (graph.id === SDXL_CANVAS_INPAINT_GRAPH) {
      if (isUsingScaledDimensions) {
        graph.edges.push({
          source: {
            node_id: MASK_RESIZE_UP,
            field: 'image',
          },
          destination: {
            node_id: SDXL_REFINER_INPAINT_CREATE_MASK,
            field: 'mask',
          },
        });
      } else {
        graph.nodes[SDXL_REFINER_INPAINT_CREATE_MASK] = {
          ...(graph.nodes[SDXL_REFINER_INPAINT_CREATE_MASK] as CreateDenoiseMaskInvocation),
          mask: canvasMaskImage,
        };
      }
    }

    if (graph.id === SDXL_CANVAS_OUTPAINT_GRAPH) {
      graph.edges.push({
        source: {
          node_id: isUsingScaledDimensions ? MASK_RESIZE_UP : MASK_COMBINE,
          field: 'image',
        },
        destination: {
          node_id: SDXL_REFINER_INPAINT_CREATE_MASK,
          field: 'mask',
        },
      });
    }

    graph.edges.push({
      source: {
        node_id: SDXL_REFINER_INPAINT_CREATE_MASK,
        field: 'denoise_mask',
      },
      destination: {
        node_id: SDXL_REFINER_DENOISE_LATENTS,
        field: 'denoise_mask',
      },
    });
  }

  if (graph.id === SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH || graph.id === SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH) {
    graph.edges.push({
      source: {
        node_id: SDXL_REFINER_DENOISE_LATENTS,
        field: 'latents',
      },
      destination: {
        node_id: isUsingScaledDimensions ? LATENTS_TO_IMAGE : CANVAS_OUTPUT,
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
};
