import type { RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import type { Graph } from 'features/nodes/util/graph/Graph';
import { getBoardField, getIsIntermediate } from 'features/nodes/util/graph/graphBuilderUtils';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import type { Invocation } from 'services/api/types';

import {
  DENOISE_LATENTS_HRF,
  ESRGAN_HRF,
  IMAGE_TO_LATENTS_HRF,
  LATENTS_TO_IMAGE_HRF_HR,
  LATENTS_TO_IMAGE_HRF_LR,
  NOISE_HRF,
  RESIZE_HRF,
} from './constants';

/**
 * Calculates the new resolution for high-resolution features (HRF) based on base model type.
 * Adjusts the width and height to maintain the aspect ratio and constrains them by the model's dimension limits,
 * rounding down to the nearest multiple of 8.
 *
 * @param {number} optimalDimension The optimal dimension for the base model.
 * @param {number} width The current width to be adjusted for HRF.
 * @param {number} height The current height to be adjusted for HRF.
 * @return {{newWidth: number, newHeight: number}} The new width and height, adjusted and rounded as needed.
 */
function calculateHrfRes(
  optimalDimension: number,
  width: number,
  height: number
): { newWidth: number; newHeight: number } {
  const aspect = width / height;

  const minDimension = Math.floor(optimalDimension * 0.5);
  const modelArea = optimalDimension * optimalDimension; // Assuming square images for model_area

  let initWidth;
  let initHeight;

  if (aspect > 1.0) {
    initHeight = Math.max(minDimension, Math.sqrt(modelArea / aspect));
    initWidth = initHeight * aspect;
  } else {
    initWidth = Math.max(minDimension, Math.sqrt(modelArea * aspect));
    initHeight = initWidth / aspect;
  }
  // Cap initial height and width to final height and width.
  initWidth = Math.min(width, initWidth);
  initHeight = Math.min(height, initHeight);

  const newWidth = roundToMultiple(Math.floor(initWidth), 8);
  const newHeight = roundToMultiple(Math.floor(initHeight), 8);

  return { newWidth, newHeight };
}

/**
 * Adds HRF to the graph.
 * @param state The root redux state
 * @param g The graph to add HRF to
 * @param denoise The denoise node
 * @param noise The noise node
 * @param l2i The l2i node
 * @param vaeSource The VAE source node (may be a model loader, VAE loader, or seamless node)
 * @returns The HRF image output node.
 */
export const addGenerationTabHRF = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  noise: Invocation<'noise'>,
  l2i: Invocation<'l2i'>,
  vaeSource: Invocation<'vae_loader'> | Invocation<'main_model_loader'> | Invocation<'seamless'>
): Invocation<'l2i'> => {
  const { hrfStrength, hrfEnabled, hrfMethod } = state.hrf;
  const { width, height } = state.controlLayers.present.size;
  const optimalDimension = selectOptimalDimension(state);
  const { newWidth: hrfWidth, newHeight: hrfHeight } = calculateHrfRes(optimalDimension, width, height);

  // Change height and width of original noise node to initial resolution.
  if (noise) {
    noise.width = hrfWidth;
    noise.height = hrfHeight;
  }

  // Define new nodes and their connections, roughly in order of operations.
  const l2iHrfLR = g.addNode({ type: 'l2i', id: LATENTS_TO_IMAGE_HRF_LR, fp32: l2i.fp32 });
  g.addEdge(denoise, 'latents', l2iHrfLR, 'latents');
  g.addEdge(vaeSource, 'vae', l2iHrfLR, 'vae');

  const resizeHrf = g.addNode({
    id: RESIZE_HRF,
    type: 'img_resize',
    width: width,
    height: height,
  });

  if (hrfMethod === 'ESRGAN') {
    let model_name: Invocation<'esrgan'>['model_name'] = 'RealESRGAN_x2plus.pth';
    if ((width * height) / (hrfWidth * hrfHeight) > 2) {
      model_name = 'RealESRGAN_x4plus.pth';
    }
    const esrganHrf = g.addNode({ id: ESRGAN_HRF, type: 'esrgan', model_name });
    g.addEdge(l2iHrfLR, 'image', esrganHrf, 'image');
    g.addEdge(esrganHrf, 'image', resizeHrf, 'image');
  } else {
    g.addEdge(l2iHrfLR, 'image', resizeHrf, 'image');
  }

  const noiseHrf = g.addNode({
    type: 'noise',
    id: NOISE_HRF,
    seed: noise.seed,
    use_cpu: noise.use_cpu,
  });
  g.addEdge(resizeHrf, 'height', noiseHrf, 'height');
  g.addEdge(resizeHrf, 'width', noiseHrf, 'width');

  const i2lHrf = g.addNode({ type: 'i2l', id: IMAGE_TO_LATENTS_HRF });
  g.addEdge(vaeSource, 'vae', i2lHrf, 'vae');
  g.addEdge(resizeHrf, 'image', i2lHrf, 'image');

  const denoiseHrf = g.addNode({
    type: 'denoise_latents',
    id: DENOISE_LATENTS_HRF,
    cfg_scale: denoise.cfg_scale,
    scheduler: denoise.scheduler,
    steps: denoise.steps,
    denoising_start: 1 - hrfStrength,
    denoising_end: 1,
  });
  g.addEdge(i2lHrf, 'latents', denoiseHrf, 'latents');
  g.addEdge(noiseHrf, 'noise', denoiseHrf, 'noise');

  // Copy edges to the original denoise into the new denoise
  g.getEdgesTo(denoise, ['control', 'ip_adapter', 'unet', 'positive_conditioning', 'negative_conditioning']).forEach(
    (edge) => {
      const clone = deepClone(edge);
      clone.destination.node_id = denoiseHrf.id;
      g.addEdgeFromObj(clone);
    }
  );

  // The original l2i node is unnecessary now, remove it
  g.deleteNode(l2i.id);

  const l2iHrfHR = g.addNode({
    type: 'l2i',
    id: LATENTS_TO_IMAGE_HRF_HR,
    fp32: l2i.fp32,
    is_intermediate: getIsIntermediate(state),
    board: getBoardField(state),
  });
  g.addEdge(vaeSource, 'vae', l2iHrfHR, 'vae');
  g.addEdge(denoiseHrf, 'latents', l2iHrfHR, 'latents');

  g.upsertMetadata({
    hrf_strength: hrfStrength,
    hrf_enabled: hrfEnabled,
    hrf_method: hrfMethod,
  });
  g.setMetadataReceivingNode(l2iHrfHR);

  return l2iHrfHR;
};
