import { logger } from 'app/logging/logger';
import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import {
  DenoiseLatentsInvocation,
  ImageResizeInvocation,
  ESRGANInvocation,
  ImageToLatentsInvocation,
  Edge,
  LatentsToImageInvocation,
  NoiseInvocation,
} from 'services/api/types';
import {
  DENOISE_LATENTS,
  NOISE,
  MAIN_MODEL_LOADER,
  LATENTS_TO_IMAGE,
  LATENTS_TO_IMAGE_HRF_HR,
  LATENTS_TO_IMAGE_HRF_LR,
  IMAGE_TO_LATENTS_HRF,
  DENOISE_LATENTS_HRF,
  RESIZE_HRF,
  ESRGAN_HRF,
  NOISE_HRF,
  VAE_LOADER,
} from './constants';
import { upsertMetadata } from './metadata';

// Copy certain connections from previous DENOISE_LATENTS to new DENOISE_LATENTS_HRF.
function copyConnectionsToDenoiseLatentsHrf(graph: NonNullableGraph): void {
  const destinationFields = [
    'vae',
    'control',
    'ip_adapter',
    'metadata',
    'unet',
    'positive_conditioning',
    'negative_conditioning',
  ];
  const newEdges: Edge[] = [];

  // Loop through the existing edges connected to DENOISE_LATENTS
  graph.edges.forEach((edge: Edge) => {
    if (
      edge.destination.node_id === DENOISE_LATENTS &&
      destinationFields.includes(edge.destination.field)
    ) {
      // Add a similar connection to DENOISE_LATENTS_HRF
      newEdges.push({
        source: {
          node_id: edge.source.node_id,
          field: edge.source.field,
        },
        destination: {
          node_id: DENOISE_LATENTS_HRF,
          field: edge.destination.field,
        },
      });
    }
  });
  graph.edges = graph.edges.concat(newEdges);
}

/**
 * Calculates the new resolution for high-resolution features (HRF) based on base model type.
 * Adjusts the width and height to maintain the aspect ratio and constrains them by the model's dimension limits,
 * rounding down to the nearest multiple of 8.
 *
 * @param {string} base_model The base model type, which determines the base dimension used in calculations.
 * @param {number} width The current width to be adjusted for HRF.
 * @param {number} height The current height to be adjusted for HRF.
 * @return {{newWidth: number, newHeight: number}} The new width and height, adjusted and rounded as needed.
 */
function calculateHrfRes(
  base_model: string,
  width: number,
  height: number
): { newWidth: number; newHeight: number } {
  const aspect = width / height;
  let dimension;
  if (base_model == 'sdxl') {
    dimension = 1024;
  } else {
    dimension = 512;
  }

  const min_dimension = Math.floor(dimension * 0.5);
  const model_area = dimension * dimension; // Assuming square images for model_area

  let init_width;
  let init_height;

  if (aspect > 1.0) {
    init_height = Math.max(min_dimension, Math.sqrt(model_area / aspect));
    init_width = init_height * aspect;
  } else {
    init_width = Math.max(min_dimension, Math.sqrt(model_area * aspect));
    init_height = init_width / aspect;
  }
  // Cap initial height and width to final height and width.
  init_width = Math.min(width, init_width);
  init_height = Math.min(height, init_height);

  const newWidth = roundToMultiple(Math.floor(init_width), 8);
  const newHeight = roundToMultiple(Math.floor(init_height), 8);

  return { newWidth, newHeight };
}

// Adds the high-res fix feature to the given graph.
export const addHrfToGraph = (
  state: RootState,
  graph: NonNullableGraph
): void => {
  // Double check hrf is enabled.
  if (
    !state.generation.hrfEnabled ||
    state.config.disabledSDFeatures.includes('hrf') ||
    state.generation.model?.model_type === 'onnx' // TODO: ONNX support
  ) {
    return;
  }
  const log = logger('txt2img');

  const { vae, hrfStrength } = state.generation;
  const isAutoVae = !vae;
  const width = state.generation.width;
  const height = state.generation.height;
  const method = state.generation.hrfMethod;
  const base_model = state.generation.model
    ? state.generation.model.base_model
    : 'sd1';
  const { newWidth: hrfWidth, newHeight: hrfHeight } = calculateHrfRes(
    base_model,
    width,
    height
  );

  // Pre-existing (original) graph nodes.
  const originalDenoiseLatentsNode = graph.nodes[DENOISE_LATENTS] as
    | DenoiseLatentsInvocation
    | undefined;
  const originalNoiseNode = graph.nodes[NOISE] as NoiseInvocation | undefined;
  const originalLatentsToImageNode = graph.nodes[LATENTS_TO_IMAGE] as
    | LatentsToImageInvocation
    | undefined;
  if (!originalDenoiseLatentsNode) {
    log.error('originalDenoiseLatentsNode is undefined');
    return;
  }
  if (!originalNoiseNode) {
    log.error('originalNoiseNode is undefined');
    return;
  }
  if (!originalLatentsToImageNode) {
    log.error('originalLatentsToImageNode is undefined');
    return;
  }

  // Change height and width of original noise node to initial resolution.
  if (originalNoiseNode) {
    originalNoiseNode.width = hrfWidth;
    originalNoiseNode.height = hrfHeight;
  }

  // Define new nodes and their connections, roughly in order of operations.
  graph.nodes[LATENTS_TO_IMAGE_HRF_LR] = {
    type: 'l2i',
    id: LATENTS_TO_IMAGE_HRF_LR,
    fp32: originalLatentsToImageNode?.fp32,
    is_intermediate: true,
  } as LatentsToImageInvocation;
  graph.edges.push(
    {
      source: {
        node_id: DENOISE_LATENTS,
        field: 'latents',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE_HRF_LR,
        field: 'latents',
      },
    },
    {
      source: {
        node_id: isAutoVae ? MAIN_MODEL_LOADER : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE_HRF_LR,
        field: 'vae',
      },
    }
  );

  graph.nodes[RESIZE_HRF] = {
    id: RESIZE_HRF,
    type: 'img_resize',
    is_intermediate: true,
    width: width,
    height: height,
  } as ImageResizeInvocation;
  if (method == 'ESRGAN') {
    let model_name = 'RealESRGAN_x2plus.pth';
    if ((width * height) / (hrfWidth * hrfHeight) > 2) {
      model_name = 'RealESRGAN_x4plus.pth';
    }
    graph.nodes[ESRGAN_HRF] = {
      id: ESRGAN_HRF,
      type: 'esrgan',
      model_name: model_name,
      is_intermediate: true,
    } as ESRGANInvocation;
    graph.edges.push(
      {
        source: {
          node_id: LATENTS_TO_IMAGE_HRF_LR,
          field: 'image',
        },
        destination: {
          node_id: ESRGAN_HRF,
          field: 'image',
        },
      },
      {
        source: {
          node_id: ESRGAN_HRF,
          field: 'image',
        },
        destination: {
          node_id: RESIZE_HRF,
          field: 'image',
        },
      }
    );
  } else {
    graph.edges.push({
      source: {
        node_id: LATENTS_TO_IMAGE_HRF_LR,
        field: 'image',
      },
      destination: {
        node_id: RESIZE_HRF,
        field: 'image',
      },
    });
  }

  graph.nodes[NOISE_HRF] = {
    type: 'noise',
    id: NOISE_HRF,
    seed: originalNoiseNode?.seed,
    use_cpu: originalNoiseNode?.use_cpu,
    is_intermediate: true,
  } as NoiseInvocation;
  graph.edges.push(
    {
      source: {
        node_id: RESIZE_HRF,
        field: 'height',
      },
      destination: {
        node_id: NOISE_HRF,
        field: 'height',
      },
    },
    {
      source: {
        node_id: RESIZE_HRF,
        field: 'width',
      },
      destination: {
        node_id: NOISE_HRF,
        field: 'width',
      },
    }
  );

  graph.nodes[IMAGE_TO_LATENTS_HRF] = {
    type: 'i2l',
    id: IMAGE_TO_LATENTS_HRF,
    is_intermediate: true,
  } as ImageToLatentsInvocation;
  graph.edges.push(
    {
      source: {
        node_id: isAutoVae ? MAIN_MODEL_LOADER : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: IMAGE_TO_LATENTS_HRF,
        field: 'vae',
      },
    },
    {
      source: {
        node_id: RESIZE_HRF,
        field: 'image',
      },
      destination: {
        node_id: IMAGE_TO_LATENTS_HRF,
        field: 'image',
      },
    }
  );

  graph.nodes[DENOISE_LATENTS_HRF] = {
    type: 'denoise_latents',
    id: DENOISE_LATENTS_HRF,
    is_intermediate: true,
    cfg_scale: originalDenoiseLatentsNode?.cfg_scale,
    scheduler: originalDenoiseLatentsNode?.scheduler,
    steps: originalDenoiseLatentsNode?.steps,
    denoising_start: 1 - state.generation.hrfStrength,
    denoising_end: 1,
  } as DenoiseLatentsInvocation;
  graph.edges.push(
    {
      source: {
        node_id: IMAGE_TO_LATENTS_HRF,
        field: 'latents',
      },
      destination: {
        node_id: DENOISE_LATENTS_HRF,
        field: 'latents',
      },
    },
    {
      source: {
        node_id: NOISE_HRF,
        field: 'noise',
      },
      destination: {
        node_id: DENOISE_LATENTS_HRF,
        field: 'noise',
      },
    }
  );
  copyConnectionsToDenoiseLatentsHrf(graph);

  graph.nodes[LATENTS_TO_IMAGE_HRF_HR] = {
    type: 'l2i',
    id: LATENTS_TO_IMAGE_HRF_HR,
    fp32: originalLatentsToImageNode?.fp32,
    is_intermediate: true,
  } as LatentsToImageInvocation;
  graph.edges.push(
    {
      source: {
        node_id: isAutoVae ? MAIN_MODEL_LOADER : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE_HRF_HR,
        field: 'vae',
      },
    },
    {
      source: {
        node_id: DENOISE_LATENTS_HRF,
        field: 'latents',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE_HRF_HR,
        field: 'latents',
      },
    }
  );
  upsertMetadata(graph, {
    hrf_strength: hrfStrength,
  });
};
