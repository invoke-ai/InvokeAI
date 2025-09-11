import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

export const buildQwenImageGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state } = arg;
  log.debug({ generationMode }, 'Building Qwen-Image graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'qwen-image', 'Selected model is not a Qwen-Image model');

  const params = selectParamsSlice(state);
  const { guidance: guidance_scale, steps: num_inference_steps, dimensions } = params;
  const prompts = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('qwen_image_graph'));

  // Nodes
  const modelLoader = g.addNode({
    type: 'qwen_image_model_loader',
    id: getPrefixedId('qwen_image_model_loader'),
    model,
    // Require explicit selection of Qwen2.5-VL in the UI
    // VAE is optional; leaving empty uses bundled VAE from the main model
  });

  const positivePrompt = g.addNode({ id: getPrefixedId('positive_prompt'), type: 'string' });

  const textEncoder = g.addNode({
    type: 'qwen_image_text_encoder',
    id: getPrefixedId('qwen_image_text_encoder'),
  });

  const negativePrompt = g.addNode({ id: getPrefixedId('negative_prompt'), type: 'string', value: prompts.negative });
  const textEncoderNeg = g.addNode({
    type: 'qwen_image_text_encoder',
    id: getPrefixedId('qwen_image_text_encoder_neg'),
  });

  const seed = g.addNode({ id: getPrefixedId('seed'), type: 'integer' });

  const denoise = g.addNode({
    type: 'qwen_image_denoise',
    id: getPrefixedId('qwen_image_denoise'),
    width: dimensions.width,
    height: dimensions.height,
    num_inference_steps,
    guidance_scale,
  });

  // Edges
  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'vae', denoise, 'vae');
  g.addEdge(modelLoader, 'qwen2_5_vl', denoise, 'qwen2_5_vl');
  g.addEdge(modelLoader, 'scheduler', denoise, 'scheduler_model');

  g.addEdge(positivePrompt, 'value', textEncoder, 'prompt');
  g.addEdge(modelLoader, 'qwen2_5_vl', textEncoder, 'qwen2_5_vl');
  g.addEdge(textEncoder, 'conditioning', denoise, 'positive_conditioning');
  g.addEdge(negativePrompt, 'value', textEncoderNeg, 'prompt');
  g.addEdge(modelLoader, 'qwen2_5_vl', textEncoderNeg, 'qwen2_5_vl');
  g.addEdge(textEncoderNeg, 'conditioning', denoise, 'negative_conditioning');

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'qwen-image');

  g.upsertMetadata({
    model: Graph.getModelMetadataField(modelConfig),
    steps: num_inference_steps,
    guidance: guidance_scale,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  let canvasOutput: Invocation | undefined = denoise;

  if (generationMode === 'txt2img') {
    g.upsertMetadata({ generation_mode: 'qwen_image_txt2img', negative_prompt: prompts.negative });
  } else if (generationMode === 'img2img') {
    // Minimal img2img path: encode image -> denoise -> image (denoise outputs image)
    // Fetch composite image from the canvas
    const adapters = arg.manager?.compositor.getVisibleAdaptersOfType('raster_layer') ?? [];
    const rect = arg.manager?.compositor.getFullCanvasBoundingBox() ?? { x: 0, y: 0, width: params.dimensions.width, height: params.dimensions.height };
    const { image_name } = await arg.manager!.compositor.getCompositeImageDTO(adapters, rect, {
      is_intermediate: true,
      silent: true,
    });

    // Resize input image to match target dims if needed
    if (dimensions.width && dimensions.height) {
      const resizeIn = g.addNode({ type: 'img_resize', id: getPrefixedId('qwen_image_i2i_resize_in'), image: { image_name }, width: dimensions.width, height: dimensions.height });
      const i2l = g.addNode({ type: 'qwen_image_i2l', id: getPrefixedId('qwen_image_i2l') });
      g.addEdge(modelLoader, 'vae', i2l, 'vae');
      g.addEdge(resizeIn, 'image', i2l, 'image');
      g.addEdge(i2l, 'latents', denoise, 'latents');
    }
    g.upsertMetadata({ generation_mode: 'qwen_image_img2img' });
  } else if (generationMode === 'inpaint') {
    // Build inpaint path similar to CogView4/FLUX
    const i2l = g.addNode({ type: 'qwen_image_i2l', id: getPrefixedId('qwen_image_i2l') });
    const seed = g.addNode({ id: getPrefixedId('seed'), type: 'integer' });
    const canvasOutput = await addInpaint({
      g,
      state,
      manager: arg.manager!,
      denoise,
      l2i: denoise,
      i2l,
      vaeSource: modelLoader,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'qwen_image_inpaint' });
    g.setMetadataReceivingNode(canvasOutput);
  } else {
    assert<Equals<typeof generationMode, never>>(false);
  }

  if (canvasOutput) {
    g.setMetadataReceivingNode(canvasOutput);
  }

  return {
    g,
    seed,
    positivePrompt,
  };
};
