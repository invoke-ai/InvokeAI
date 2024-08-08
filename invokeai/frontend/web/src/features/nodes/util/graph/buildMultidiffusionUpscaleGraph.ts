import type { RootState } from 'app/store/store';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import type { GraphType } from 'features/nodes/util/graph/generation/Graph';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { isNonRefinerMainModelConfig, isSpandrelImageToImageModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import {
  CLIP_SKIP,
  CONTROL_NET_COLLECT,
  IMAGE_TO_LATENTS,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  SDXL_MODEL_LOADER,
  SPANDREL,
  TILED_MULTI_DIFFUSION_DENOISE_LATENTS,
  UNSHARP_MASK,
  VAE_LOADER,
} from './constants';
import { addLoRAs } from './generation/addLoRAs';
import { addSDXLLoRas } from './generation/addSDXLLoRAs';
import { getBoardField, getPresetModifiedPrompts } from './graphBuilderUtils';

export const buildMultidiffusionUpscaleGraph = async (state: RootState): Promise<GraphType> => {
  const { model, cfgScale: cfg_scale, scheduler, steps, vaePrecision, seed, vae } = state.generation;
  const { upscaleModel, upscaleInitialImage, structure, creativity, tileControlnetModel, scale } = state.upscale;

  assert(model, 'No model found in state');
  assert(upscaleModel, 'No upscale model found in state');
  assert(upscaleInitialImage, 'No initial image found in state');
  assert(tileControlnetModel, 'Tile controlnet is required');

  const g = new Graph();

  const upscaleNode = g.addNode({
    id: SPANDREL,
    type: 'spandrel_image_to_image_autoscale',
    image: upscaleInitialImage,
    image_to_image_model: upscaleModel,
    fit_to_multiple_of_8: true,
    scale,
  });

  const unsharpMaskNode2 = g.addNode({
    id: `${UNSHARP_MASK}_2`,
    type: 'unsharp_mask',
    radius: 2,
    strength: 60,
  });

  g.addEdge(upscaleNode, 'image', unsharpMaskNode2, 'image');

  const noiseNode = g.addNode({
    id: NOISE,
    type: 'noise',
    seed,
  });

  g.addEdge(unsharpMaskNode2, 'width', noiseNode, 'width');
  g.addEdge(unsharpMaskNode2, 'height', noiseNode, 'height');

  const i2lNode = g.addNode({
    id: IMAGE_TO_LATENTS,
    type: 'i2l',
    fp32: vaePrecision === 'fp32',
    tiled: true,
  });

  g.addEdge(unsharpMaskNode2, 'image', i2lNode, 'image');

  const l2iNode = g.addNode({
    type: 'l2i',
    id: LATENTS_TO_IMAGE,
    fp32: vaePrecision === 'fp32',
    tiled: true,
    board: getBoardField(state),
    is_intermediate: false,
  });

  const tiledMultidiffusionNode = g.addNode({
    id: TILED_MULTI_DIFFUSION_DENOISE_LATENTS,
    type: 'tiled_multi_diffusion_denoise_latents',
    tile_height: 1024, // is this dependent on base model
    tile_width: 1024, // is this dependent on base model
    tile_overlap: 128,
    steps,
    cfg_scale,
    scheduler,
    denoising_start: ((creativity * -1 + 10) * 4.99) / 100,
    denoising_end: 1,
  });

  let posCondNode;
  let negCondNode;
  let modelNode;

  if (model.base === 'sdxl') {
    const { positivePrompt, negativePrompt, positiveStylePrompt, negativeStylePrompt } =
      getPresetModifiedPrompts(state);

    posCondNode = g.addNode({
      type: 'sdxl_compel_prompt',
      id: POSITIVE_CONDITIONING,
      prompt: positivePrompt,
      style: positiveStylePrompt,
    });
    negCondNode = g.addNode({
      type: 'sdxl_compel_prompt',
      id: NEGATIVE_CONDITIONING,
      prompt: negativePrompt,
      style: negativeStylePrompt,
    });
    modelNode = g.addNode({
      type: 'sdxl_model_loader',
      id: SDXL_MODEL_LOADER,
      model,
    });
    g.addEdge(modelNode, 'clip', posCondNode, 'clip');
    g.addEdge(modelNode, 'clip', negCondNode, 'clip');
    g.addEdge(modelNode, 'clip2', posCondNode, 'clip2');
    g.addEdge(modelNode, 'clip2', negCondNode, 'clip2');
    g.addEdge(modelNode, 'unet', tiledMultidiffusionNode, 'unet');
    addSDXLLoRas(state, g, tiledMultidiffusionNode, modelNode, null, posCondNode, negCondNode);

    const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);

    g.upsertMetadata({
      cfg_scale,
      positive_prompt: positivePrompt,
      negative_prompt: negativePrompt,
      positive_style_prompt: positiveStylePrompt,
      negative_style_prompt: negativeStylePrompt,
      model: Graph.getModelMetadataField(modelConfig),
      seed,
      steps,
      scheduler,
      vae: vae ?? undefined,
    });
  } else {
    const { positivePrompt, negativePrompt } = getPresetModifiedPrompts(state);

    posCondNode = g.addNode({
      type: 'compel',
      id: POSITIVE_CONDITIONING,
      prompt: positivePrompt,
    });
    negCondNode = g.addNode({
      type: 'compel',
      id: NEGATIVE_CONDITIONING,
      prompt: negativePrompt,
    });
    modelNode = g.addNode({
      type: 'main_model_loader',
      id: MAIN_MODEL_LOADER,
      model,
    });
    const clipSkipNode = g.addNode({
      type: 'clip_skip',
      id: CLIP_SKIP,
    });

    g.addEdge(modelNode, 'clip', clipSkipNode, 'clip');
    g.addEdge(clipSkipNode, 'clip', posCondNode, 'clip');
    g.addEdge(clipSkipNode, 'clip', negCondNode, 'clip');
    g.addEdge(modelNode, 'unet', tiledMultidiffusionNode, 'unet');
    addLoRAs(state, g, tiledMultidiffusionNode, modelNode, null, clipSkipNode, posCondNode, negCondNode);

    const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
    const upscaleModelConfig = await fetchModelConfigWithTypeGuard(upscaleModel.key, isSpandrelImageToImageModelConfig);

    g.upsertMetadata({
      cfg_scale,
      positive_prompt: positivePrompt,
      negative_prompt: negativePrompt,
      model: Graph.getModelMetadataField(modelConfig),
      seed,
      steps,
      scheduler,
      vae: vae ?? undefined,
      upscale_model: Graph.getModelMetadataField(upscaleModelConfig),
      creativity,
      structure,
    });
  }

  g.setMetadataReceivingNode(l2iNode);
  g.addEdgeToMetadata(upscaleNode, 'width', 'width');
  g.addEdgeToMetadata(upscaleNode, 'height', 'height');

  let vaeNode;
  if (vae) {
    vaeNode = g.addNode({
      id: VAE_LOADER,
      type: 'vae_loader',
      vae_model: vae,
    });
  }

  g.addEdge(vaeNode || modelNode, 'vae', i2lNode, 'vae');
  g.addEdge(vaeNode || modelNode, 'vae', l2iNode, 'vae');

  g.addEdge(noiseNode, 'noise', tiledMultidiffusionNode, 'noise');
  g.addEdge(i2lNode, 'latents', tiledMultidiffusionNode, 'latents');
  g.addEdge(posCondNode, 'conditioning', tiledMultidiffusionNode, 'positive_conditioning');
  g.addEdge(negCondNode, 'conditioning', tiledMultidiffusionNode, 'negative_conditioning');

  g.addEdge(tiledMultidiffusionNode, 'latents', l2iNode, 'latents');

  const controlnetNode1 = g.addNode({
    id: 'controlnet_1',
    type: 'controlnet',
    control_model: tileControlnetModel,
    control_mode: 'balanced',
    resize_mode: 'just_resize',
    control_weight: (structure + 10) * 0.0325 + 0.3,
    begin_step_percent: 0,
    end_step_percent: (structure + 10) * 0.025 + 0.3,
  });

  g.addEdge(unsharpMaskNode2, 'image', controlnetNode1, 'image');

  const controlnetNode2 = g.addNode({
    id: 'controlnet_2',
    type: 'controlnet',
    control_model: tileControlnetModel,
    control_mode: 'balanced',
    resize_mode: 'just_resize',
    control_weight: ((structure + 10) * 0.0325 + 0.15) * 0.45,
    begin_step_percent: (structure + 10) * 0.025 + 0.3,
    end_step_percent: 0.85,
  });

  g.addEdge(unsharpMaskNode2, 'image', controlnetNode2, 'image');

  const collectNode = g.addNode({
    id: CONTROL_NET_COLLECT,
    type: 'collect',
  });
  g.addEdge(controlnetNode1, 'control', collectNode, 'item');
  g.addEdge(controlnetNode2, 'control', collectNode, 'item');

  g.addEdge(collectNode, 'collection', tiledMultidiffusionNode, 'control');

  return g.getGraph();
};
