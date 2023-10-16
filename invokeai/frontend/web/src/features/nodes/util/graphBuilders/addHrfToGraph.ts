import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import {
  DenoiseLatentsInvocation,
  NoiseInvocation,
  LatentsToImageInvocation,
  ImageResizeInvocation,
  Edge,
  ImageToLatentsInvocation,
} from 'services/api/types';
import {
  LATENTS_TO_IMAGE,
  DENOISE_LATENTS,
  NOISE,
  MAIN_MODEL_LOADER,
  METADATA_ACCUMULATOR,
  LATENTS_TO_IMAGE_HRF_HR,
  LATENTS_TO_IMAGE_HRF_LR,
  IMAGE_TO_LATENTS_HRF,
  DENOISE_LATENTS_HRF,
  RESIZE_HRF,
  NOISE_HRF,
  VAE_LOADER,
} from './constants';
import { logger } from 'app/logging/logger';

// To recap, next steps for this feature might include:

// Automatic calculation of initial dimensions
// Support ESRGAN & PIL image upscaling to allow for lower denoising strength & to retain original image composition
// Add control adapters used for txt2img phase to img2img phase

// I want to capture that the recommended way now is to convert latents to an image, upscale that image, then convert the image back to latents

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

  const { vae } = state.generation;
  const isAutoVae = !vae;
  const hrfWidth = state.generation.hrfWidth;
  const hrfHeight = state.generation.hrfHeight;

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
    width: state.generation.width,
    height: state.generation.height,
  } as ImageResizeInvocation;
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
    },
    {
      source: {
        node_id: METADATA_ACCUMULATOR,
        field: 'metadata',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE_HRF_HR,
        field: 'metadata',
      },
    }
  );
};
