import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addZImageLoRAs } from 'features/nodes/util/graph/generation/addZImageLoRAs';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { isNonRefinerMainModelConfig, isSpandrelImageToImageModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { getBoardField } from './graphBuilderUtils';
import type { GraphBuilderReturn } from './types';

export const buildZImageMultidiffusionUpscaleGraph = async (state: RootState): Promise<GraphBuilderReturn> => {
  const { model, steps, zImageScheduler, zImageVaeModel, zImageQwen3EncoderModel, zImageQwen3SourceModel } =
    state.params;
  const {
    upscaleModel,
    upscaleInitialImage,
    structure,
    creativity,
    tileControlnetModel,
    scale,
    tileSize,
    tileOverlap,
  } = state.upscale;

  assert(model, 'No model selected');
  assert(model.base === 'z-image', 'Z-Image upscaling requires a Z-Image model');
  assert(upscaleModel, 'No upscale model found in state');
  assert(upscaleInitialImage, 'No initial image found in state');

  const g = new Graph();

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });

  // Step 1: Spandrel upscale
  const spandrelAutoscale = g.addNode({
    type: 'spandrel_image_to_image_autoscale',
    id: getPrefixedId('spandrel_autoscale'),
    image: upscaleInitialImage,
    image_to_image_model: upscaleModel,
    fit_to_multiple_of_8: true,
    scale,
  });

  // Step 2: Unsharp mask
  const unsharpMask = g.addNode({
    type: 'unsharp_mask',
    id: getPrefixedId('unsharp_2'),
    radius: 2,
    strength: 60,
  });
  g.addEdge(spandrelAutoscale, 'image', unsharpMask, 'image');

  // Step 3: Z-Image model loader
  const modelLoader = g.addNode({
    type: 'z_image_model_loader',
    id: getPrefixedId('z_image_model_loader'),
    model,
    vae_model: zImageVaeModel ?? undefined,
    qwen3_encoder_model: zImageQwen3EncoderModel ?? undefined,
    qwen3_source_model: zImageQwen3SourceModel ?? undefined,
  });

  // Step 4: Z-Image text encoder (positive)
  const posCond = g.addNode({
    type: 'z_image_text_encoder',
    id: getPrefixedId('z_image_text_encoder_pos'),
  });
  g.addEdge(modelLoader, 'qwen3_encoder', posCond, 'qwen3_encoder');
  g.addEdge(positivePrompt, 'value', posCond, 'prompt');

  // Step 5: Z-Image VAE encode
  const zImageI2L = g.addNode({
    type: 'z_image_i2l',
    id: getPrefixedId('z_image_i2l'),
  });
  g.addEdge(unsharpMask, 'image', zImageI2L, 'image');
  g.addEdge(modelLoader, 'vae', zImageI2L, 'vae');

  // Step 6: Seed
  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });

  // Step 7: Tiled Z-Image denoise
  const tiledDenoise = g.addNode({
    type: 'tiled_z_image_denoise',
    id: getPrefixedId('tiled_z_image_denoise'),
    tile_height: tileSize,
    tile_width: tileSize,
    tile_overlap: tileOverlap,
    steps,
    scheduler: zImageScheduler,
    guidance_scale: 1.0, // Z-Image Turbo works best without CFG
    denoising_start: ((creativity * -1 + 10) * 4.99) / 100,
    denoising_end: 1,
  });

  g.addEdge(zImageI2L, 'latents', tiledDenoise, 'latents');
  g.addEdge(seed, 'value', tiledDenoise, 'seed');
  g.addEdge(modelLoader, 'transformer', tiledDenoise, 'transformer');
  g.addEdge(posCond, 'conditioning', tiledDenoise, 'positive_conditioning');

  // Connect width/height from upscaled image
  g.addEdge(unsharpMask, 'width', tiledDenoise, 'width');
  g.addEdge(unsharpMask, 'height', tiledDenoise, 'height');

  // Step 8: Z-Image VAE decode
  const zImageL2I = g.addNode({
    type: 'z_image_l2i',
    id: getPrefixedId('z_image_l2i'),
    board: getBoardField(state),
    is_intermediate: false,
  });
  g.addEdge(tiledDenoise, 'latents', zImageL2I, 'latents');
  g.addEdge(modelLoader, 'vae', zImageL2I, 'vae');

  // Step 9: ControlNet for tile guidance (if available)
  if (tileControlnetModel) {
    const zImageControl = g.addNode({
      id: 'z_image_control_1',
      type: 'z_image_control',
      control_model: tileControlnetModel,
      control_context_scale: (structure + 10) * 0.0325 + 0.3,
      begin_step_percent: 0,
      end_step_percent: (structure + 10) * 0.025 + 0.3,
    });
    g.addEdge(unsharpMask, 'image', zImageControl, 'image');
    g.addEdge(zImageControl, 'control', tiledDenoise, 'control');
    // VAE is needed for encoding control images
    g.addEdge(modelLoader, 'vae', tiledDenoise, 'vae');
  }

  // Step 10: Add LoRAs
  addZImageLoRAs(state, g, tiledDenoise, modelLoader, posCond, null);

  // Metadata
  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  const upscaleModelConfig = await fetchModelConfigWithTypeGuard(upscaleModel.key, isSpandrelImageToImageModelConfig);

  g.upsertMetadata({
    model: Graph.getModelMetadataField(modelConfig),
    steps,
    upscale_model: Graph.getModelMetadataField(upscaleModelConfig),
    creativity,
    structure,
    tile_size: tileSize,
    tile_overlap: tileOverlap,
    upscale_initial_image: {
      image_name: upscaleInitialImage.image_name,
      width: upscaleInitialImage.width,
      height: upscaleInitialImage.height,
    },
    upscale_scale: scale,
  });

  g.setMetadataReceivingNode(zImageL2I);
  g.addEdgeToMetadata(spandrelAutoscale, 'width', 'width');
  g.addEdgeToMetadata(spandrelAutoscale, 'height', 'height');
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  return {
    g,
    seed,
    positivePrompt,
  };
};
