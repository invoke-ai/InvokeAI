import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addSDXLLoRAs } from 'features/nodes/util/graph/generation/addSDXLLoRAs';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { isNonRefinerMainModelConfig, isSpandrelImageToImageModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { addLoRAs } from './generation/addLoRAs';
import { getBoardField, getPresetModifiedPrompts } from './graphBuilderUtils';

export const buildMultidiffusionUpscaleGraph = async (state: RootState): Promise<Graph> => {
  const { model, cfgScale: cfg_scale, scheduler, steps, vaePrecision, seed, vae } = state.canvasV2.params;
  const { upscaleModel, upscaleInitialImage, structure, creativity, tileControlnetModel, scale } = state.upscale;

  assert(model, 'No model found in state');
  assert(upscaleModel, 'No upscale model found in state');
  assert(upscaleInitialImage, 'No initial image found in state');
  assert(tileControlnetModel, 'Tile controlnet is required');

  const g = new Graph();

  const upscaleNode = g.addNode({
    type: 'spandrel_image_to_image_autoscale',
    id: getPrefixedId('spandrel_autoscale'),
    image: upscaleInitialImage,
    image_to_image_model: upscaleModel,
    fit_to_multiple_of_8: true,
    scale,
  });

  const unsharpMaskNode2 = g.addNode({
    type: 'unsharp_mask',
    id: getPrefixedId('unsharp_2'),
    radius: 2,
    strength: 60,
  });

  g.addEdge(upscaleNode, 'image', unsharpMaskNode2, 'image');

  const noiseNode = g.addNode({
    type: 'noise',
    id: getPrefixedId('noise'),
    seed,
  });

  g.addEdge(unsharpMaskNode2, 'width', noiseNode, 'width');
  g.addEdge(unsharpMaskNode2, 'height', noiseNode, 'height');

  const i2lNode = g.addNode({
    type: 'i2l',
    id: getPrefixedId('i2l'),
    fp32: vaePrecision === 'fp32',
    tiled: true,
  });

  g.addEdge(unsharpMaskNode2, 'image', i2lNode, 'image');

  const l2iNode = g.addNode({
    type: 'l2i',
    id: getPrefixedId('l2i'),
    fp32: vaePrecision === 'fp32',
    tiled: true,
    board: getBoardField(state),
    is_intermediate: false,
  });

  const tiledMultidiffusionNode = g.addNode({
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

  let posCondNode;
  let negCondNode;
  let modelNode;

  if (model.base === 'sdxl') {
    const { positivePrompt, negativePrompt, positiveStylePrompt, negativeStylePrompt } =
      getPresetModifiedPrompts(state);

    posCondNode = g.addNode({
      type: 'sdxl_compel_prompt',
      id: getPrefixedId('pos_cond'),
      prompt: positivePrompt,
      style: positiveStylePrompt,
    });
    negCondNode = g.addNode({
      type: 'sdxl_compel_prompt',
      id: getPrefixedId('neg_cond'),
      prompt: negativePrompt,
      style: negativeStylePrompt,
    });
    modelNode = g.addNode({
      type: 'sdxl_model_loader',
      id: getPrefixedId('sdxl_model_loader'),
      model,
    });
    g.addEdge(modelNode, 'clip', posCondNode, 'clip');
    g.addEdge(modelNode, 'clip', negCondNode, 'clip');
    g.addEdge(modelNode, 'clip2', posCondNode, 'clip2');
    g.addEdge(modelNode, 'clip2', negCondNode, 'clip2');
    g.addEdge(modelNode, 'unet', tiledMultidiffusionNode, 'unet');
    addSDXLLoRAs(state, g, tiledMultidiffusionNode, modelNode, null, posCondNode, negCondNode);

    g.upsertMetadata({
      positive_prompt: positivePrompt,
      negative_prompt: negativePrompt,
      positive_style_prompt: positiveStylePrompt,
      negative_style_prompt: negativeStylePrompt,
    });
  } else {
    const { positivePrompt, negativePrompt } = getPresetModifiedPrompts(state);

    posCondNode = g.addNode({
      type: 'compel',
      id: getPrefixedId('pos_cond'),
      prompt: positivePrompt,
    });
    negCondNode = g.addNode({
      type: 'compel',
      id: getPrefixedId('neg_cond'),
      prompt: negativePrompt,
    });
    modelNode = g.addNode({
      type: 'main_model_loader',
      id: getPrefixedId('sd1_model_loader'),
      model,
    });
    const clipSkipNode = g.addNode({
      type: 'clip_skip',
      id: getPrefixedId('clip_skip'),
    });

    g.addEdge(modelNode, 'clip', clipSkipNode, 'clip');
    g.addEdge(clipSkipNode, 'clip', posCondNode, 'clip');
    g.addEdge(clipSkipNode, 'clip', negCondNode, 'clip');
    g.addEdge(modelNode, 'unet', tiledMultidiffusionNode, 'unet');
    addLoRAs(state, g, tiledMultidiffusionNode, modelNode, null, clipSkipNode, posCondNode, negCondNode);

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

  g.setMetadataReceivingNode(l2iNode);
  g.addEdgeToMetadata(upscaleNode, 'width', 'width');
  g.addEdgeToMetadata(upscaleNode, 'height', 'height');

  let vaeNode;
  if (vae) {
    vaeNode = g.addNode({
      type: 'vae_loader',
      id: getPrefixedId('vae'),
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
    type: 'collect',
    id: getPrefixedId('controlnet_collector'),
  });
  g.addEdge(controlnetNode1, 'control', collectNode, 'item');
  g.addEdge(controlnetNode2, 'control', collectNode, 'item');

  g.addEdge(collectNode, 'collection', tiledMultidiffusionNode, 'control');

  return g;
};
