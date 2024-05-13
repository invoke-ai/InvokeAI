import type { RootState } from 'app/store/store';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import type { Graph } from 'features/nodes/util/graph/Graph';
import { MetadataUtil } from 'features/nodes/util/graph/MetadataUtil';
import type { Invocation } from 'services/api/types';
import { isRefinerMainModelModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import {
  SDXL_REFINER_DENOISE_LATENTS,
  SDXL_REFINER_MODEL_LOADER,
  SDXL_REFINER_NEGATIVE_CONDITIONING,
  SDXL_REFINER_POSITIVE_CONDITIONING,
  SDXL_REFINER_SEAMLESS,
} from './constants';
import { getModelMetadataField } from './metadata';

export const addGenerationTabSDXLRefiner = async (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  modelLoader: Invocation<'sdxl_model_loader'>,
  seamless: Invocation<'seamless'> | null,
  posCond: Invocation<'sdxl_compel_prompt'>,
  negCond: Invocation<'sdxl_compel_prompt'>,
  l2i: Invocation<'l2i'>
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

  assert(refinerModel, 'No refiner model found in state');

  const modelConfig = await fetchModelConfigWithTypeGuard(refinerModel.key, isRefinerMainModelModelConfig);

  // We need to re-route latents to the refiner
  g.deleteEdgesFrom(denoise, ['latents']);
  // Latents will now come from refiner - delete edges to the l2i VAE decode
  g.deleteEdgesTo(l2i, ['latents']);

  const refinerModelLoader = g.addNode({
    type: 'sdxl_refiner_model_loader',
    id: SDXL_REFINER_MODEL_LOADER,
    model: refinerModel,
  });
  const refinerPosCond = g.addNode({
    type: 'sdxl_refiner_compel_prompt',
    id: SDXL_REFINER_POSITIVE_CONDITIONING,
    style: posCond.style,
    aesthetic_score: refinerPositiveAestheticScore,
  });
  const refinerNegCond = g.addNode({
    type: 'sdxl_refiner_compel_prompt',
    id: SDXL_REFINER_NEGATIVE_CONDITIONING,
    style: negCond.style,
    aesthetic_score: refinerNegativeAestheticScore,
  });
  const refinerDenoise = g.addNode({
    type: 'denoise_latents',
    id: SDXL_REFINER_DENOISE_LATENTS,
    cfg_scale: refinerCFGScale,
    steps: refinerSteps,
    scheduler: refinerScheduler,
    denoising_start: refinerStart,
    denoising_end: 1,
  });

  if (seamless) {
    const refinerSeamless = g.addNode({
      id: SDXL_REFINER_SEAMLESS,
      type: 'seamless',
      seamless_x: seamless.seamless_x,
      seamless_y: seamless.seamless_y,
    });
    g.addEdge(refinerModelLoader, 'unet', refinerSeamless, 'unet');
    g.addEdge(refinerModelLoader, 'vae', refinerSeamless, 'vae');
    g.addEdge(refinerSeamless, 'unet', refinerDenoise, 'unet');
  } else {
    g.addEdge(refinerModelLoader, 'unet', refinerDenoise, 'unet');
  }

  g.addEdge(refinerModelLoader, 'clip2', refinerPosCond, 'clip2');
  g.addEdge(refinerModelLoader, 'clip2', refinerNegCond, 'clip2');
  g.addEdge(refinerPosCond, 'conditioning', refinerDenoise, 'positive_conditioning');
  g.addEdge(refinerNegCond, 'conditioning', refinerDenoise, 'negative_conditioning');
  g.addEdge(denoise, 'latents', refinerDenoise, 'latents');
  g.addEdge(refinerDenoise, 'latents', l2i, 'latents');

  MetadataUtil.add(g, {
    refiner_model: getModelMetadataField(modelConfig),
    refiner_positive_aesthetic_score: refinerPositiveAestheticScore,
    refiner_negative_aesthetic_score: refinerNegativeAestheticScore,
    refiner_cfg_scale: refinerCFGScale,
    refiner_scheduler: refinerScheduler,
    refiner_start: refinerStart,
    refiner_steps: refinerSteps,
  });
};
