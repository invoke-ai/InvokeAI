import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import {
  DenoiseLatentsInvocation,
  RescaleLatentsInvocation,
  ONNXTextToLatentsInvocation,
  NoiseInvocation,
} from 'services/api/types';
import { LATENTS_TO_IMAGE, NSFW_CHECKER } from './constants';

// Adds high-res fix to the given graph by
// adding an additional denoise latents with the same parameters
// but with an upscaled version of the original latents.
export const addHrfToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  denoiseLatentsId: string,
  noiseId: string
): void => {
  const DENOISE_LATENTS_HRF = `${denoiseLatentsId}_hrf`;
  const RESCALE_LATENTS = 'rescale_latents';
  const NOISE_HRF = 'noise_hrf';

  const originalDenoiseLatentsNode = graph.nodes[denoiseLatentsId] as
    | DenoiseLatentsInvocation
    | ONNXTextToLatentsInvocation
    | undefined;

  const originalNoiseNode = graph.nodes[noiseId] as NoiseInvocation;
  const isUsingOnnxDenoise = originalDenoiseLatentsNode?.type == 't2l_onnx';

  // Scale by hrfScale.
  const hrfScale = state.generation.hrfScale;
  const scaledHeight = originalNoiseNode?.height
    ? originalNoiseNode.height * hrfScale
    : undefined;
  const scaledWidth = originalNoiseNode?.width
    ? originalNoiseNode.width * hrfScale
    : undefined;

  // New nodes
  // Denoise latents node to be run on upscaled latents.
  const hrfT2lNode: DenoiseLatentsInvocation | ONNXTextToLatentsInvocation =
    isUsingOnnxDenoise
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
          denoising_start: 0,
          denoising_end: 1,
        };

  const rescaleLatentsNode: RescaleLatentsInvocation = {
    id: RESCALE_LATENTS,
    type: 'lresize',
  };

  const hrfNoise: NoiseInvocation = {
    type: 'noise',
    id: NOISE_HRF,
    seed: originalNoiseNode.seed,
    width: scaledWidth,
    height: scaledHeight,
    use_cpu: originalNoiseNode.use_cpu,
    is_intermediate: originalNoiseNode.is_intermediate,
  };
};
