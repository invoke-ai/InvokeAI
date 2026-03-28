import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import { isQwenImageReferenceImageConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { zImageField } from 'features/nodes/types/common';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addQwenImageLoRAs } from 'features/nodes/util/graph/generation/addQwenImageLoRAs';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
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

export const buildQwenImageGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Qwen Image Edit graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'qwen-image', 'Selected model is not a Qwen Image Edit model');

  const params = selectParamsSlice(state);

  const { cfgScale: cfg_scale, steps } = params;

  const prompts = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('qwen_image_graph'));

  const modelLoader = g.addNode({
    type: 'qwen_image_model_loader',
    id: getPrefixedId('qwen_image_model_loader'),
    model,
    component_source: params.qwenImageComponentSource,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'qwen_image_text_encoder',
    id: getPrefixedId('pos_prompt'),
    quantization: params.qwenImageQuantization,
  });

  // Negative conditioning with a blank prompt for CFG
  const negCond = g.addNode({
    type: 'qwen_image_text_encoder',
    id: getPrefixedId('neg_prompt'),
    prompt: prompts.negative || ' ',
    quantization: params.qwenImageQuantization,
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  const denoise = g.addNode({
    type: 'qwen_image_denoise',
    id: getPrefixedId('denoise_latents'),
    cfg_scale,
    steps,
    shift: params.qwenImageShift,
  });
  const l2i = g.addNode({
    type: 'qwen_image_l2i',
    id: getPrefixedId('l2i'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'qwen_vl_encoder', posCond, 'qwen_vl_encoder');
  g.addEdge(modelLoader, 'qwen_vl_encoder', negCond, 'qwen_vl_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(posCond, 'conditioning', denoise, 'positive_conditioning');

  g.addEdge(negCond, 'conditioning', denoise, 'negative_conditioning');

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  // Add Qwen Image Edit LoRAs if any are enabled
  addQwenImageLoRAs(state, g, denoise, modelLoader);

  // Only collect reference images for edit-variant models.
  // For txt2img (generate) models, reference images are not used even if they exist in state.
  const isEditModel = 'variant' in model && model.variant === 'edit';
  const validRefImageConfigs = isEditModel
    ? selectRefImagesSlice(state).entities.filter(
        (entity) =>
          entity.isEnabled &&
          isQwenImageReferenceImageConfig(entity.config) &&
          entity.config.image !== null &&
          getGlobalReferenceImageWarnings(entity, model).length === 0
      )
    : [];

  if (validRefImageConfigs.length > 0) {
    const refImgCollect = g.addNode({
      type: 'collect',
      id: getPrefixedId('qwen_ref_img_collect'),
    });
    for (const { config } of validRefImageConfigs) {
      const imgField = zImageField.parse(config.image?.crop?.image ?? config.image?.original.image);
      const imageNode = g.addNode({
        type: 'image',
        id: getPrefixedId('qwen_ref_img'),
        image: imgField,
      });
      g.addEdge(imageNode, 'image', refImgCollect, 'item');
    }
    // Pass reference images to text encoder for vision-language conditioning
    g.addEdge(refImgCollect, 'collection', posCond, 'reference_images');

    // Also VAE-encode the first reference image as latents for the denoising transformer.
    // The transformer expects [noisy_patches ; ref_patches] in its sequence.
    const firstConfig = validRefImageConfigs[0]!;
    const firstImgField = zImageField.parse(
      firstConfig.config.image?.crop?.image ?? firstConfig.config.image?.original.image
    );
    // Don't force-resize the reference image to the output dimensions — that would
    // distort the aspect ratio when they differ. The I2L encodes at the image's
    // native size; the denoise node handles dimension mismatches via interpolation.
    const refI2l = g.addNode({
      type: 'qwen_image_i2l',
      id: getPrefixedId('qwen_ref_i2l'),
    });
    const refImageNode = g.addNode({
      type: 'image',
      id: getPrefixedId('qwen_ref_img_for_vae'),
      image: firstImgField,
    });
    g.addEdge(refImageNode, 'image', refI2l, 'image');
    g.addEdge(modelLoader, 'vae', refI2l, 'vae');
    g.addEdge(refI2l, 'latents', denoise, 'reference_latents');

    g.upsertMetadata({ ref_images: validRefImageConfigs }, 'merge');
  }

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'qwen-image');

  g.upsertMetadata({
    cfg_scale,
    negative_prompt: prompts.negative,
    model: Graph.getModelMetadataField(modelConfig),
    qwen_image_component_source: params.qwenImageComponentSource,
    qwen_image_quantization: params.qwenImageQuantization,
    qwen_image_shift: params.qwenImageShift,
    steps,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage({
      g,
      state,
      denoise,
      l2i,
    });
    g.upsertMetadata({ generation_mode: 'qwen_image_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'qwen_image_i2l',
      id: getPrefixedId('qwen_image_i2l'),
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
    g.upsertMetadata({ generation_mode: 'qwen_image_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'qwen_image_i2l',
      id: getPrefixedId('qwen_image_i2l'),
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
    g.upsertMetadata({ generation_mode: 'qwen_image_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'qwen_image_i2l',
      id: getPrefixedId('qwen_image_i2l'),
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
    g.upsertMetadata({ generation_mode: 'qwen_image_outpaint' });
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
