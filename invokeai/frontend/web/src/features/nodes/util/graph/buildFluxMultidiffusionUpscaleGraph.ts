import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addFLUXLoRAs } from 'features/nodes/util/graph/generation/addFLUXLoRAs';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { isNonRefinerMainModelConfig, isSpandrelImageToImageModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { getBoardField } from './graphBuilderUtils';
import type { GraphBuilderReturn } from './types';

export const buildFluxMultidiffusionUpscaleGraph = async (state: RootState): Promise<GraphBuilderReturn> => {
  const { model, steps, guidance: fluxGuidance, fluxScheduler, t5EncoderModel, clipEmbedModel, fluxVAE } = state.params;
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
  assert(model.base === 'flux', 'FLUX upscaling requires a FLUX model');
  assert(upscaleModel, 'No upscale model found in state');
  assert(upscaleInitialImage, 'No initial image found in state');
  assert(t5EncoderModel, 'No T5 Encoder model found in state');
  assert(clipEmbedModel, 'No CLIP Embed model found in state');
  assert(fluxVAE, 'No FLUX VAE model found in state');

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

  // Step 2: Unsharp mask for sharpening
  const unsharpMask = g.addNode({
    type: 'unsharp_mask',
    id: getPrefixedId('unsharp_2'),
    radius: 2,
    strength: 60,
  });
  g.addEdge(spandrelAutoscale, 'image', unsharpMask, 'image');

  // Step 3: FLUX model loader
  const modelLoader = g.addNode({
    type: 'flux_model_loader',
    id: getPrefixedId('flux_model_loader'),
    model,
    t5_encoder_model: t5EncoderModel,
    clip_embed_model: clipEmbedModel,
    vae_model: fluxVAE,
  });

  // Step 4: FLUX text encoder
  const fluxTextEncoder = g.addNode({
    type: 'flux_text_encoder',
    id: getPrefixedId('flux_text_encoder'),
  });
  g.addEdge(modelLoader, 'clip', fluxTextEncoder, 'clip');
  g.addEdge(modelLoader, 't5_encoder', fluxTextEncoder, 't5_encoder');
  g.addEdge(modelLoader, 'max_seq_len', fluxTextEncoder, 't5_max_seq_len');
  g.addEdge(positivePrompt, 'value', fluxTextEncoder, 'prompt');

  // Step 5: FLUX VAE encode (image to latents)
  const fluxVaeEncode = g.addNode({
    type: 'flux_vae_encode',
    id: getPrefixedId('flux_vae_encode'),
  });
  g.addEdge(unsharpMask, 'image', fluxVaeEncode, 'image');
  g.addEdge(modelLoader, 'vae', fluxVaeEncode, 'vae');

  // Step 6: Seed
  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });

  // Step 7: Tiled FLUX denoise
  const tiledFluxDenoise = g.addNode({
    type: 'tiled_flux_denoise',
    id: getPrefixedId('tiled_flux_denoise'),
    tile_height: tileSize,
    tile_width: tileSize,
    tile_overlap: tileOverlap,
    num_steps: steps,
    guidance: fluxGuidance,
    scheduler: fluxScheduler,
    denoising_start: ((creativity * -1 + 10) * 4.99) / 100,
    denoising_end: 1,
  });

  g.addEdge(fluxVaeEncode, 'latents', tiledFluxDenoise, 'latents');
  g.addEdge(seed, 'value', tiledFluxDenoise, 'seed');
  g.addEdge(modelLoader, 'transformer', tiledFluxDenoise, 'transformer');
  g.addEdge(modelLoader, 'vae', tiledFluxDenoise, 'controlnet_vae');

  // Connect width/height from upscaled image
  g.addEdge(unsharpMask, 'width', tiledFluxDenoise, 'width');
  g.addEdge(unsharpMask, 'height', tiledFluxDenoise, 'height');

  // Connect text conditioning (via collect node)
  const posCondCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('pos_cond_collect'),
  });
  g.addEdge(fluxTextEncoder, 'conditioning', posCondCollect, 'item');
  g.addEdge(posCondCollect, 'collection', tiledFluxDenoise, 'positive_text_conditioning');

  // Step 8: FLUX VAE decode (latents to image)
  const fluxVaeDecode = g.addNode({
    type: 'flux_vae_decode',
    id: getPrefixedId('flux_vae_decode'),
    board: getBoardField(state),
    is_intermediate: false,
  });
  g.addEdge(tiledFluxDenoise, 'latents', fluxVaeDecode, 'latents');
  g.addEdge(modelLoader, 'vae', fluxVaeDecode, 'vae');

  // Step 9: ControlNet for tile guidance (if available)
  if (tileControlnetModel) {
    const controlNet1 = g.addNode({
      id: 'flux_controlnet_1',
      type: 'flux_controlnet',
      control_model: tileControlnetModel,
      control_weight: (structure + 10) * 0.0325 + 0.3,
      begin_step_percent: 0,
      end_step_percent: (structure + 10) * 0.025 + 0.3,
    });
    g.addEdge(unsharpMask, 'image', controlNet1, 'image');

    const controlNet2 = g.addNode({
      id: 'flux_controlnet_2',
      type: 'flux_controlnet',
      control_model: tileControlnetModel,
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
    g.addEdge(controlNetCollector, 'collection', tiledFluxDenoise, 'control');
  }

  // Step 10: Add LoRAs
  addFLUXLoRAs(state, g, tiledFluxDenoise, modelLoader, fluxTextEncoder);

  // Metadata
  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  const upscaleModelConfig = await fetchModelConfigWithTypeGuard(upscaleModel.key, isSpandrelImageToImageModelConfig);

  g.upsertMetadata({
    guidance: fluxGuidance,
    model: Graph.getModelMetadataField(modelConfig),
    steps,
    vae: fluxVAE ?? undefined,
    t5_encoder: t5EncoderModel ?? undefined,
    clip_embed_model: clipEmbedModel ?? undefined,
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

  g.setMetadataReceivingNode(fluxVaeDecode);
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
