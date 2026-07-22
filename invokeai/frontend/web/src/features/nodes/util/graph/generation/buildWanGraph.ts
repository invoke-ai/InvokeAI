import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import { isWanReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { zImageField } from 'features/nodes/types/common';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWanLoRAs } from 'features/nodes/util/graph/generation/addWanLoRAs';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields, selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

/**
 * Build a graph for Wan 2.2 image generation.
 *
 * Phase 9 piece #1: text-to-image only, Diffusers main model with all
 * components (transformer, VAE, UMT5-XXL encoder) resolved from the main
 * model itself. Subsequent pieces will add:
 *   - img2img (Latents input + Image-to-Latents wiring + denoising_start)
 *   - I2V (ref-image encoder, A14B I2V variant gate)
 *   - LoRAs (single + collection)
 *   - Inpaint (mask handling)
 *   - Standalone VAE / T5 / GGUF low-noise-expert wiring via params slice
 */
export const buildWanGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Wan 2.2 graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'wan', 'Selected model is not a Wan model');

  // Fetch the full config early so we can branch on variant. I2V flows
  // route the raster image through wan_ref_image_encoder instead of
  // wan_i2l, so the variant has to be known before we choose a graph
  // shape — not after.
  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'wan');
  const isI2V = modelConfig.variant === 'i2v_a14b';

  const params = selectParamsSlice(state);
  const { cfgScale: cfg_scale, steps } = params;
  const prompts = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('wan_graph'));

  const modelLoader = g.addNode({
    type: 'wan_model_loader',
    id: getPrefixedId('wan_model_loader'),
    model,
    transformer_low_noise_model: params.wanTransformerLowNoise ?? undefined,
    component_source: params.wanComponentSource ?? undefined,
    vae_model: params.wanVaeModel ?? undefined,
    wan_t5_encoder_model: params.wanT5EncoderModel ?? undefined,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'wan_text_encoder',
    id: getPrefixedId('pos_prompt'),
  });

  // CFG is mathematically inactive at scale 1.0 — skip the negative branch
  // entirely so each step runs only one forward pass.
  const useCfg = cfg_scale > 1;
  const negCond = useCfg
    ? g.addNode({
        type: 'wan_text_encoder',
        id: getPrefixedId('neg_prompt'),
        prompt: prompts.negative || ' ',
      })
    : null;

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });

  const denoise = g.addNode({
    type: 'wan_denoise',
    id: getPrefixedId('denoise_latents'),
    guidance_scale: cfg_scale,
    // The denoise node treats values < 1.0 (including the FE's default 0) as
    // "fall back to the primary guidance_scale". Forward null/undefined when
    // the user hasn't set an explicit low-noise CFG so the backend handles it.
    guidance_scale_low_noise: params.wanGuidanceScaleLowNoise ?? undefined,
    steps,
  });

  const l2i = g.addNode({
    type: 'wan_l2i',
    id: getPrefixedId('l2i'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'wan_t5_encoder', posCond, 'wan_t5_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(posCond, 'conditioning', denoise, 'positive_conditioning');

  if (negCond) {
    g.addEdge(modelLoader, 'wan_t5_encoder', negCond, 'wan_t5_encoder');
    g.addEdge(negCond, 'conditioning', denoise, 'negative_conditioning');
  }

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  // Wan LoRAs (high-noise, low-noise, and untagged). The collection loader
  // is inserted between modelLoader and denoise; both expert routing and
  // dual-list population happen on the backend based on each LoRA's
  // recorded ``expert`` tag. The helper also filters out variant-incompatible
  // LoRAs (e.g. A14B Lightning on a TI2V-5B main) so the layer patcher
  // doesn't crash on a shape mismatch.
  await addWanLoRAs(state, g, denoise, modelLoader, modelConfig);

  g.upsertMetadata({
    cfg_scale,
    negative_prompt: prompts.negative,
    model: Graph.getModelMetadataField(modelConfig),
    steps,
    wan_transformer_low_noise: params.wanTransformerLowNoise,
    wan_component_source: params.wanComponentSource,
    wan_vae_model: params.wanVaeModel,
    wan_t5_encoder_model: params.wanT5EncoderModel,
    wan_guidance_scale_low_noise: params.wanGuidanceScaleLowNoise,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  // I2V variants take a reference image from the global Reference Images
  // panel (same UX as Qwen Image Edit / FLUX.2 Klein). The image is encoded
  // by the model's own VAE and concatenated to the noise latents along the
  // channel dim each step (transformer in_channels=36 on I2V). Canvas modes
  // (img2img/inpaint/outpaint) don't apply to I2V — the ref image fully
  // replaces what a raster layer used to provide.
  if (isI2V) {
    assert(
      generationMode === 'txt2img',
      'Wan 2.2 I2V only supports text-to-image with a reference image. ' +
        'Use a T2V or TI2V model for canvas img2img / inpaint / outpaint.'
    );

    const wanRefEntity = selectRefImagesSlice(state).entities.find(
      (entity) =>
        entity.isEnabled &&
        isWanReferenceImageConfig(entity.config) &&
        entity.config.image !== null &&
        getGlobalReferenceImageWarnings(entity, modelConfig).length === 0
    );
    assert(
      wanRefEntity && isWanReferenceImageConfig(wanRefEntity.config) && wanRefEntity.config.image,
      'Wan 2.2 I2V requires a reference image. Add one in the Reference Images panel.'
    );

    canvasOutput = addTextToImage({ g, state, denoise, l2i });
    const refImageField = zImageField.parse(
      wanRefEntity.config.image.crop?.image ?? wanRefEntity.config.image.original.image
    );
    const refEncoder = g.addNode({
      type: 'wan_ref_image_encoder',
      id: getPrefixedId('wan_ref_encoder'),
      image: refImageField,
      width: denoise.width,
      height: denoise.height,
    });
    g.addEdge(modelLoader, 'vae', refEncoder, 'vae');
    g.addEdge(refEncoder, 'ref_image', denoise, 'ref_image');

    g.upsertMetadata({ generation_mode: 'wan_i2v' });
  } else if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage({
      g,
      state,
      denoise,
      l2i,
    });
    g.upsertMetadata({ generation_mode: 'wan_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'wan_i2l',
      id: getPrefixedId('wan_i2l'),
    });
    canvasOutput = await addImageToImage({
      g,
      state,
      manager,
      denoise,
      l2i,
      i2l,
      vaeSource: modelLoader,
    });
    g.upsertMetadata({ generation_mode: 'wan_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'wan_i2l',
      id: getPrefixedId('wan_i2l'),
    });
    canvasOutput = await addInpaint({
      g,
      state,
      manager,
      l2i,
      i2l,
      denoise,
      vaeSource: modelLoader,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'wan_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'wan_i2l',
      id: getPrefixedId('wan_i2l'),
    });
    canvasOutput = await addOutpaint({
      g,
      state,
      manager,
      l2i,
      i2l,
      denoise,
      vaeSource: modelLoader,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'wan_outpaint' });
  } else {
    assert<Equals<typeof generationMode, never>>(false);
  }

  if (state.system.shouldUseNSFWChecker) {
    canvasOutput = addNSFWChecker(g, canvasOutput);
  }
  if (state.system.shouldUseWatermarker) {
    canvasOutput = addWatermarker(g, canvasOutput);
  }

  g.updateNode(canvasOutput, selectCanvasOutputFields(state));

  if (selectActiveTab(state) === 'canvas') {
    g.upsertMetadata(selectCanvasMetadata(state));
  }

  g.setMetadataReceivingNode(canvasOutput);

  return {
    g,
    seed,
    positivePrompt,
  };
};
