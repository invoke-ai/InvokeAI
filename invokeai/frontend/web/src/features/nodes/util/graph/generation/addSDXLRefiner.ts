import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation } from 'services/api/types';
import { isRefinerMainModelModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

export const addSDXLRefiner = async (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
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
  } = state.params;

  assert(refinerModel, 'No refiner model found in state');

  const modelConfig = await fetchModelConfigWithTypeGuard(refinerModel.key, isRefinerMainModelModelConfig);

  // We need to re-route latents to the refiner
  g.deleteEdgesFrom(denoise, ['latents']);
  // Latents will now come from refiner - delete edges to the l2i VAE decode
  g.deleteEdgesTo(l2i, ['latents']);

  const refinerModelLoader = g.addNode({
    type: 'sdxl_refiner_model_loader',
    id: getPrefixedId('refiner_model_loader'),
    model: refinerModel,
  });
  const refinerPosCond = g.addNode({
    type: 'sdxl_refiner_compel_prompt',
    id: getPrefixedId('refiner_pos_cond'),
    style: posCond.style,
    aesthetic_score: refinerPositiveAestheticScore,
  });
  const refinerNegCond = g.addNode({
    type: 'sdxl_refiner_compel_prompt',
    id: getPrefixedId('refiner_neg_cond'),
    style: negCond.style,
    aesthetic_score: refinerNegativeAestheticScore,
  });
  const refinerDenoise = g.addNode({
    type: 'denoise_latents',
    id: getPrefixedId('refiner_denoise_latents'),
    cfg_scale: refinerCFGScale,
    steps: refinerSteps,
    scheduler: refinerScheduler,
    denoising_start: refinerStart,
    denoising_end: 1,
  });

  if (seamless) {
    const refinerSeamless = g.addNode({
      type: 'seamless',
      id: getPrefixedId('refiner_seamless'),
      seamless_x: seamless.seamless_x,
      seamless_y: seamless.seamless_y,
    });
    g.addEdge(refinerModelLoader, 'unet', refinerSeamless, 'unet');
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

  g.upsertMetadata({
    refiner_model: Graph.getModelMetadataField(modelConfig),
    refiner_positive_aesthetic_score: refinerPositiveAestheticScore,
    refiner_negative_aesthetic_score: refinerNegativeAestheticScore,
    refiner_cfg_scale: refinerCFGScale,
    refiner_scheduler: refinerScheduler,
    refiner_start: refinerStart,
    refiner_steps: refinerSteps,
  });
};
