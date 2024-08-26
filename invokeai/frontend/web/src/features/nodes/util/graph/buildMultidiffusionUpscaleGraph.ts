import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addSDXLLoRAs } from 'features/nodes/util/graph/generation/addSDXLLoRAs';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig, isSpandrelImageToImageModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { addLoRAs } from './generation/addLoRAs';
import { getBoardField, getPresetModifiedPrompts } from './graphBuilderUtils';

export const buildMultidiffusionUpscaleGraph = async (
  state: RootState
): Promise<{ g: Graph; noise: Invocation<'noise'>; posCond: Invocation<'compel' | 'sdxl_compel_prompt'> }> => {
  const { model, cfgScale: cfg_scale, scheduler, steps, vaePrecision, seed, vae } = state.params;
  const { upscaleModel, upscaleInitialImage, structure, creativity, tileControlnetModel, scale } = state.upscale;

  assert(model, 'No model found in state');
  assert(upscaleModel, 'No upscale model found in state');
  assert(upscaleInitialImage, 'No initial image found in state');
  assert(tileControlnetModel, 'Tile controlnet is required');

  const g = new Graph();

  const spandrelAutoscale = g.addNode({
    type: 'spandrel_image_to_image_autoscale',
    id: getPrefixedId('spandrel_autoscale'),
    image: upscaleInitialImage,
    image_to_image_model: upscaleModel,
    fit_to_multiple_of_8: true,
    scale,
  });

  const unsharpMask = g.addNode({
    type: 'unsharp_mask',
    id: getPrefixedId('unsharp_2'),
    radius: 2,
    strength: 60,
  });

  g.addEdge(spandrelAutoscale, 'image', unsharpMask, 'image');

  const noise = g.addNode({
    type: 'noise',
    id: getPrefixedId('noise'),
    seed,
  });

  g.addEdge(unsharpMask, 'width', noise, 'width');
  g.addEdge(unsharpMask, 'height', noise, 'height');

  const i2l = g.addNode({
    type: 'i2l',
    id: getPrefixedId('i2l'),
    fp32: vaePrecision === 'fp32',
    tiled: true,
  });

  g.addEdge(unsharpMask, 'image', i2l, 'image');

  const l2i = g.addNode({
    type: 'l2i',
    id: getPrefixedId('l2i'),
    fp32: vaePrecision === 'fp32',
    tiled: true,
    board: getBoardField(state),
    is_intermediate: false,
  });

  const tiledMultidiffusion = g.addNode({
    type: 'tiled_multi_diffusion_denoise_latents',
    id: getPrefixedId('tiled_multidiffusion_denoise_latents'),
    tile_height: 1024, // is this dependent on base model
    tile_width: 1024, // is this dependent on base model
    tile_overlap: 128,
    steps,
    cfg_scale,
    scheduler,
    denoising_start: ((creativity * -1 + 10) * 4.99) / 100,
    denoising_end: 1,
  });

  let posCond;
  let negCond;
  let modelLoader;

  if (model.base === 'sdxl') {
    const { positivePrompt, negativePrompt, positiveStylePrompt, negativeStylePrompt } =
      getPresetModifiedPrompts(state);

    posCond = g.addNode({
      type: 'sdxl_compel_prompt',
      id: getPrefixedId('pos_cond'),
      prompt: positivePrompt,
      style: positiveStylePrompt,
    });
    negCond = g.addNode({
      type: 'sdxl_compel_prompt',
      id: getPrefixedId('neg_cond'),
      prompt: negativePrompt,
      style: negativeStylePrompt,
    });
    modelLoader = g.addNode({
      type: 'sdxl_model_loader',
      id: getPrefixedId('sdxl_model_loader'),
      model,
    });
    g.addEdge(modelLoader, 'clip', posCond, 'clip');
    g.addEdge(modelLoader, 'clip', negCond, 'clip');
    g.addEdge(modelLoader, 'clip2', posCond, 'clip2');
    g.addEdge(modelLoader, 'clip2', negCond, 'clip2');
    g.addEdge(modelLoader, 'unet', tiledMultidiffusion, 'unet');
    addSDXLLoRAs(state, g, tiledMultidiffusion, modelLoader, null, posCond, negCond);

    g.upsertMetadata({
      positive_prompt: positivePrompt,
      negative_prompt: negativePrompt,
      positive_style_prompt: positiveStylePrompt,
      negative_style_prompt: negativeStylePrompt,
    });
  } else {
    const { positivePrompt, negativePrompt } = getPresetModifiedPrompts(state);

    posCond = g.addNode({
      type: 'compel',
      id: getPrefixedId('pos_cond'),
      prompt: positivePrompt,
    });
    negCond = g.addNode({
      type: 'compel',
      id: getPrefixedId('neg_cond'),
      prompt: negativePrompt,
    });
    modelLoader = g.addNode({
      type: 'main_model_loader',
      id: getPrefixedId('sd1_model_loader'),
      model,
    });
    const clipSkipNode = g.addNode({
      type: 'clip_skip',
      id: getPrefixedId('clip_skip'),
    });

    g.addEdge(modelLoader, 'clip', clipSkipNode, 'clip');
    g.addEdge(clipSkipNode, 'clip', posCond, 'clip');
    g.addEdge(clipSkipNode, 'clip', negCond, 'clip');
    g.addEdge(modelLoader, 'unet', tiledMultidiffusion, 'unet');
    addLoRAs(state, g, tiledMultidiffusion, modelLoader, null, clipSkipNode, posCond, negCond);

    g.upsertMetadata({
      positive_prompt: positivePrompt,
      negative_prompt: negativePrompt,
    });
  }

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  const upscaleModelConfig = await fetchModelConfigWithTypeGuard(upscaleModel.key, isSpandrelImageToImageModelConfig);

  g.upsertMetadata({
    cfg_scale,
    model: Graph.getModelMetadataField(modelConfig),
    seed,
    steps,
    scheduler,
    vae: vae ?? undefined,
    upscale_model: Graph.getModelMetadataField(upscaleModelConfig),
    creativity,
    structure,
    upscale_initial_image: {
      image_name: upscaleInitialImage.image_name,
      width: upscaleInitialImage.width,
      height: upscaleInitialImage.height,
    },
    upscale_scale: scale,
  });

  g.setMetadataReceivingNode(l2i);
  g.addEdgeToMetadata(spandrelAutoscale, 'width', 'width');
  g.addEdgeToMetadata(spandrelAutoscale, 'height', 'height');

  let vaeLoader;
  if (vae) {
    vaeLoader = g.addNode({
      type: 'vae_loader',
      id: getPrefixedId('vae'),
      vae_model: vae,
    });
  }

  g.addEdge(vaeLoader || modelLoader, 'vae', i2l, 'vae');
  g.addEdge(vaeLoader || modelLoader, 'vae', l2i, 'vae');

  g.addEdge(noise, 'noise', tiledMultidiffusion, 'noise');
  g.addEdge(i2l, 'latents', tiledMultidiffusion, 'latents');
  g.addEdge(posCond, 'conditioning', tiledMultidiffusion, 'positive_conditioning');
  g.addEdge(negCond, 'conditioning', tiledMultidiffusion, 'negative_conditioning');

  g.addEdge(tiledMultidiffusion, 'latents', l2i, 'latents');

  const controlNet1 = g.addNode({
    id: 'controlnet_1',
    type: 'controlnet',
    control_model: tileControlnetModel,
    control_mode: 'balanced',
    resize_mode: 'just_resize',
    control_weight: (structure + 10) * 0.0325 + 0.3,
    begin_step_percent: 0,
    end_step_percent: (structure + 10) * 0.025 + 0.3,
  });

  g.addEdge(unsharpMask, 'image', controlNet1, 'image');

  const controlNet2 = g.addNode({
    id: 'controlnet_2',
    type: 'controlnet',
    control_model: tileControlnetModel,
    control_mode: 'balanced',
    resize_mode: 'just_resize',
    control_weight: ((structure + 10) * 0.0325 + 0.15) * 0.45,
    begin_step_percent: (structure + 10) * 0.025 + 0.3,
    end_step_percent: 0.85,
  });

  g.addEdge(unsharpMask, 'image', controlNet2, 'image');

  const controlNetCollector = g.addNode({
    type: 'collect',
    id: getPrefixedId('controlnet_collector'),
  });
  g.addEdge(controlNet1, 'control', controlNetCollector, 'item');
  g.addEdge(controlNet2, 'control', controlNetCollector, 'item');

  g.addEdge(controlNetCollector, 'collection', tiledMultidiffusion, 'control');

  return { g, noise, posCond };
};
