import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import {
  DenoiseLatentsInvocation,
  RescaleLatentsInvocation,
  ONNXTextToLatentsInvocation,
  NoiseInvocation,
  LatentsToImageInvocation,
} from 'services/api/types';
import {
  LATENTS_TO_IMAGE,
  POSITIVE_CONDITIONING,
  NEGATIVE_CONDITIONING,
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

// Adds high-res fix to the given graph by
// adding an additional denoise latents with the same parameters
// but with an upscaled version of the original latents.
export const addHrfToGraph = (
  state: RootState,
  graph: NonNullableGraph
): void => {
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
  const scaledHeight = originalNoiseNode?.height
    ? originalNoiseNode.height * hrfScale
    : undefined;
  const scaledWidth = originalNoiseNode?.width
    ? originalNoiseNode.width * hrfScale
    : undefined;

  // Define new nodes.
  // Denoise latents node to be run on upscaled latents.
  // Define new nodes.
  const denoiseLatentsHrfNode: DenoiseLatentsInvocation = {
    type: 'denoise_latents',
    id: DENOISE_LATENTS_HRF,
    is_intermediate: originalDenoiseLatentsNode?.is_intermediate,
    cfg_scale: originalDenoiseLatentsNode?.cfg_scale,
    scheduler: originalDenoiseLatentsNode?.scheduler,
    steps: originalDenoiseLatentsNode?.steps,
    denoising_start: state.generation.hrfStrength,
    denoising_end: 1,
  };

  const rescaleLatentsNode: RescaleLatentsInvocation = {
    id: RESCALE_LATENTS,
    type: 'lresize',
    width: scaledWidth,
    height: scaledHeight,
  };

  const hrfNoiseNode: NoiseInvocation = {
    type: 'noise',
    id: NOISE_HRF,
    seed: originalNoiseNode.seed,
    width: scaledWidth,
    height: scaledHeight,
    use_cpu: originalNoiseNode.use_cpu,
    is_intermediate: originalNoiseNode.is_intermediate,
  };

  const latentsToImageHrfNode: LatentsToImageInvocation = {
    type: originalLatentsToImageNode.type,
    id: LATENTS_TO_IMAGE_HRF,
    vae: originalLatentsToImageNode.vae,
    fp32: originalLatentsToImageNode.fp32,
    is_intermediate: originalLatentsToImageNode.is_intermediate,
  };

  // Add new noes to graph.
  graph.nodes[LATENTS_TO_IMAGE_HRF] =
    latentsToImageHrfNode as LatentsToImageInvocation;
  graph.nodes[DENOISE_LATENTS_HRF] =
    denoiseLatentsHrfNode as DenoiseLatentsInvocation;
  graph.nodes[RESCALE_LATENTS] = rescaleLatentsNode as RescaleLatentsInvocation;
  graph.nodes[NOISE_HRF] = hrfNoiseNode as NoiseInvocation;

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
    {
      source: {
        node_id: MAIN_MODEL_LOADER,
        field: 'unet',
      },
      destination: {
        node_id: DENOISE_LATENTS_HRF,
        field: 'unet',
      },
    },
    {
      source: {
        node_id: POSITIVE_CONDITIONING,
        field: 'conditioning',
      },
      destination: {
        node_id: DENOISE_LATENTS_HRF,
        field: 'positive_conditioning',
      },
    },
    {
      source: {
        node_id: NEGATIVE_CONDITIONING,
        field: 'conditioning',
      },
      destination: {
        node_id: DENOISE_LATENTS_HRF,
        field: 'negative_conditioning',
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
};
