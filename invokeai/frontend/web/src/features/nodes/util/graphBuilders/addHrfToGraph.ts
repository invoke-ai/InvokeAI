import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import {
  DenoiseLatentsInvocation,
  ResizeLatentsInvocation,
  NoiseInvocation,
  LatentsToImageInvocation,
  Edge,
} from 'services/api/types';
import {
  LATENTS_TO_IMAGE,
  DENOISE_LATENTS,
  NOISE,
  MAIN_MODEL_LOADER,
  METADATA_ACCUMULATOR,
  LATENTS_TO_IMAGE_HRF,
  DENOISE_LATENTS_HRF,
  RESCALE_LATENTS,
  NOISE_HRF,
  VAE_LOADER,
} from './constants';
import { logger } from 'app/logging/logger';

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
  if (!state.generation.hrfEnabled) {
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
  // Original latents to image should pick this up.
  const originalLatentsToImageNode = graph.nodes[LATENTS_TO_IMAGE] as
    | LatentsToImageInvocation
    | undefined;
  // Check if originalDenoiseLatentsNode is undefined and log an error
  if (!originalDenoiseLatentsNode) {
    log.error('originalDenoiseLatentsNode is undefined');
    return;
  }
  // Check if originalNoiseNode is undefined and log an error
  if (!originalNoiseNode) {
    log.error('originalNoiseNode is undefined');
    return;
  }

  // Check if originalLatentsToImageNode is undefined and log an error
  if (!originalLatentsToImageNode) {
    log.error('originalLatentsToImageNode is undefined');
    return;
  }
  // Change height and width of original noise node to initial resolution.
  if (originalNoiseNode) {
    originalNoiseNode.width = hrfWidth;
    originalNoiseNode.height = hrfHeight;
  }

  // Define new nodes.
  // Denoise latents node to be run on upscaled latents.
  const denoiseLatentsHrfNode: DenoiseLatentsInvocation = {
    type: 'denoise_latents',
    id: DENOISE_LATENTS_HRF,
    is_intermediate: originalDenoiseLatentsNode?.is_intermediate,
    cfg_scale: originalDenoiseLatentsNode?.cfg_scale,
    scheduler: originalDenoiseLatentsNode?.scheduler,
    steps: originalDenoiseLatentsNode?.steps,
    denoising_start: 1 - state.generation.hrfStrength,
    denoising_end: 1,
  };

  // New base resolution noise node.
  const hrfNoiseNode: NoiseInvocation = {
    type: 'noise',
    id: NOISE_HRF,
    seed: originalNoiseNode?.seed,
    use_cpu: originalNoiseNode?.use_cpu,
    is_intermediate: originalNoiseNode?.is_intermediate,
  };

  const rescaleLatentsNode: ResizeLatentsInvocation = {
    id: RESCALE_LATENTS,
    type: 'lresize',
    width: state.generation.width,
    height: state.generation.height,
  };

  // New node to convert latents to image.
  const latentsToImageHrfNode: LatentsToImageInvocation | undefined =
    originalLatentsToImageNode
      ? {
          type: 'l2i',
          id: LATENTS_TO_IMAGE_HRF,
          fp32: originalLatentsToImageNode?.fp32,
          is_intermediate: originalLatentsToImageNode?.is_intermediate,
        }
      : undefined;

  // Add new nodes to graph.
  graph.nodes[LATENTS_TO_IMAGE_HRF] =
    latentsToImageHrfNode as LatentsToImageInvocation;
  graph.nodes[DENOISE_LATENTS_HRF] =
    denoiseLatentsHrfNode as DenoiseLatentsInvocation;
  graph.nodes[NOISE_HRF] = hrfNoiseNode as NoiseInvocation;
  graph.nodes[RESCALE_LATENTS] = rescaleLatentsNode as ResizeLatentsInvocation;

  // Connect nodes.
  graph.edges.push(
    {
      // Set up rescale latents.
      source: {
        node_id: DENOISE_LATENTS,
        field: 'latents',
      },
      destination: {
        node_id: RESCALE_LATENTS,
        field: 'latents',
      },
    },
    // Set up new noise node
    {
      source: {
        node_id: RESCALE_LATENTS,
        field: 'height',
      },
      destination: {
        node_id: NOISE_HRF,
        field: 'height',
      },
    },
    {
      source: {
        node_id: RESCALE_LATENTS,
        field: 'width',
      },
      destination: {
        node_id: NOISE_HRF,
        field: 'width',
      },
    },
    // Set up new denoise node.
    {
      source: {
        node_id: RESCALE_LATENTS,
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
    },
    // Set up new latents to image node.
    {
      source: {
        node_id: DENOISE_LATENTS_HRF,
        field: 'latents',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE_HRF,
        field: 'latents',
      },
    },
    {
      source: {
        node_id: METADATA_ACCUMULATOR,
        field: 'metadata',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE_HRF,
        field: 'metadata',
      },
    },
    {
      source: {
        node_id: isAutoVae ? MAIN_MODEL_LOADER : VAE_LOADER,
        field: 'vae',
      },
      destination: {
        node_id: LATENTS_TO_IMAGE_HRF,
        field: 'vae',
      },
    }
  );

  copyConnectionsToDenoiseLatentsHrf(graph);
};
