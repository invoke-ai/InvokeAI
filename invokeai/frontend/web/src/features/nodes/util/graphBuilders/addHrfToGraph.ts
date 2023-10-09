import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import {
  DenoiseLatentsInvocation,
  ResizeLatentsInvocation,
  MetadataAccumulatorInvocation,
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

  const { vae } = state.generation;
  const isAutoVae = !vae;

  // Pre-existing (original) graph nodes.
  const originalDenoiseLatentsNode = graph.nodes[
    DENOISE_LATENTS
  ] as DenoiseLatentsInvocation;
  const originalNoiseNode = graph.nodes[NOISE] as NoiseInvocation;
  const originalLatentsToImageNode = graph.nodes[
    LATENTS_TO_IMAGE
  ] as LatentsToImageInvocation;

  // Scale height and width by hrfScale.
  const hrfScale = state.generation.hrfScale;
  const scaledHeight = state.generation.height * hrfScale;
  const scaledWidth = state.generation.width * hrfScale;

  // Add hrf information to the metadata accumulator.
  const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
    | MetadataAccumulatorInvocation
    | undefined;
  if (metadataAccumulator) {
    metadataAccumulator.height = scaledHeight;
    metadataAccumulator.width = scaledWidth;
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

  const rescaleLatentsNode: ResizeLatentsInvocation = {
    id: RESCALE_LATENTS,
    type: 'lresize',
    width: scaledWidth,
    height: scaledHeight,
  };

  // New higher resolution noise node.
  const hrfNoiseNode: NoiseInvocation = {
    type: 'noise',
    id: NOISE_HRF,
    seed: originalNoiseNode.seed,
    width: scaledWidth,
    height: scaledHeight,
    use_cpu: originalNoiseNode.use_cpu,
    is_intermediate: originalNoiseNode.is_intermediate,
  };

  // New node to convert latents to image.
  const latentsToImageHrfNode: LatentsToImageInvocation | undefined =
    originalLatentsToImageNode
      ? {
          type: originalLatentsToImageNode.type,
          id: LATENTS_TO_IMAGE_HRF,
          vae: originalLatentsToImageNode.vae,
          fp32: originalLatentsToImageNode.fp32,
          is_intermediate: originalLatentsToImageNode.is_intermediate,
        }
      : undefined;

  // Add new nodes to graph.
  graph.nodes[LATENTS_TO_IMAGE_HRF] =
    latentsToImageHrfNode as LatentsToImageInvocation;

  graph.nodes[DENOISE_LATENTS_HRF] = denoiseLatentsHrfNode as
    | DenoiseLatentsInvocation
    | undefined;
  graph.nodes[RESCALE_LATENTS] = rescaleLatentsNode as
    | ResizeLatentsInvocation
    | undefined;
  graph.nodes[NOISE_HRF] = hrfNoiseNode as NoiseInvocation | undefined;

  // Connect nodes.
  graph.edges.push(
    {
      // Set up resize latents.
      source: {
        node_id: DENOISE_LATENTS,
        field: 'latents',
      },
      destination: {
        node_id: RESCALE_LATENTS,
        field: 'latents',
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
