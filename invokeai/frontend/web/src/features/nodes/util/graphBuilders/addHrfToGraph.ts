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
  NSFW_CHECKER,
  VAE_LOADER,
  POSITIVE_CONDITIONING,
  NEGATIVE_CONDITIONING,
  DENOISE_LATENTS,
  NOISE,
  MAIN_MODEL_LOADER,
} from './constants';

// Adds high-res fix to the given graph by
// adding an additional denoise latents with the same parameters
// but with an upscaled version of the original latents.
export const addHrfToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  isUsingOnnxModel: boolean
): void => {
  const originalDenoiseLatentsNode = graph.nodes[DENOISE_LATENTS] as
    | DenoiseLatentsInvocation
    | ONNXTextToLatentsInvocation
    | undefined;
  const originalNoiseNode = graph.nodes[NOISE] as NoiseInvocation;
  const originalLatentsToImageNode = graph.nodes[
    LATENTS_TO_IMAGE
  ] as LatentsToImageInvocation;

  // Scale by hrfScale.
  const hrfScale = state.generation.hrfScale;
  const scaledHeight = originalNoiseNode?.height
    ? originalNoiseNode.height * hrfScale
    : undefined;
  const scaledWidth = originalNoiseNode?.width
    ? originalNoiseNode.width * hrfScale
    : undefined;

  // New nodes
  const LATENTS_TO_IMAGE_HRF = 'latents_to_image_hrf';
  const latentsToImageNode: LatentsToImageInvocation = {
    type: originalLatentsToImageNode.type,
    id: LATENTS_TO_IMAGE_HRF,
    vae: originalLatentsToImageNode.vae,
    fp32: originalLatentsToImageNode.fp32,
    is_intermediate: originalLatentsToImageNode.is_intermediate,
  };

  // Denoise latents node to be run on upscaled latents.
  const DENOISE_LATENTS_HRF = `${DENOISE_LATENTS}_hrf`;
  const denoiseLatentsHrfNode:
    | DenoiseLatentsInvocation
    | ONNXTextToLatentsInvocation = isUsingOnnxModel
    ? {
        type: 't2l_onnx',
        id: DENOISE_LATENTS_HRF,
        is_intermediate: originalDenoiseLatentsNode?.is_intermediate,
        cfg_scale: originalDenoiseLatentsNode?.cfg_scale,
        scheduler: originalDenoiseLatentsNode?.scheduler,
        steps: originalDenoiseLatentsNode?.steps,
      }
    : {
        type: 'denoise_latents',
        id: DENOISE_LATENTS_HRF,
        is_intermediate: originalDenoiseLatentsNode?.is_intermediate,
        cfg_scale: originalDenoiseLatentsNode?.cfg_scale,
        scheduler: originalDenoiseLatentsNode?.scheduler,
        steps: originalDenoiseLatentsNode?.steps,
        // TODO: Make this customizable.
        denoising_start: 0.5,
        denoising_end: 1,
      };

  const RESCALE_LATENTS = 'rescale_latents';
  const rescaleLatentsNode: RescaleLatentsInvocation = {
    id: RESCALE_LATENTS,
    type: 'lresize',
    width: scaledWidth,
    height: scaledHeight,
  };

  const NOISE_HRF = 'noise_hrf';
  const hrfNoiseNode: NoiseInvocation = {
    type: 'noise',
    id: NOISE_HRF,
    seed: originalNoiseNode.seed,
    width: scaledWidth,
    height: scaledHeight,
    use_cpu: originalNoiseNode.use_cpu,
    is_intermediate: originalNoiseNode.is_intermediate,
  };

  // Connect nodes
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
        field: 'positive_conditioning',
      },
      destination: {
        node_id: DENOISE_LATENTS_HRF,
        field: 'positive_conditioning',
      },
    },
    {
      source: {
        node_id: NEGATIVE_CONDITIONING,
        field: 'negative_conditioning',
      },
      destination: {
        node_id: DENOISE_LATENTS_HRF,
        field: 'negative_conditioning',
      },
    },
    // Set up new latents to image node.
    {
      source: {
        node_id: NEGATIVE_CONDITIONING,
        field: 'negative_conditioning',
      },
      destination: {
        node_id: DENOISE_LATENTS_HRF,
        field: 'negative_conditioning',
      },
    }
  );
};
