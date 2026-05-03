import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  selectErnieImageScheduler,
  selectErnieImageUsePromptEnhancer,
  selectMainModelConfig,
  selectParamsSlice,
} from 'features/controlLayers/store/paramsSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
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

export const buildErnieImageGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building ERNIE-Image graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'ernie-image', 'Selected model is not an ERNIE-Image model');

  const params = selectParamsSlice(state);
  const { cfgScale: guidance_scale, steps } = params;
  const ernieImageScheduler = selectErnieImageScheduler(state);
  const usePromptEnhancer = selectErnieImageUsePromptEnhancer(state);

  const prompts = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('ernie_image_graph'));

  const modelLoader = g.addNode({
    type: 'ernie_image_model_loader',
    id: getPrefixedId('ernie_image_model_loader'),
    model,
    use_prompt_enhancer: usePromptEnhancer,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });

  const posCond = g.addNode({
    type: 'ernie_image_text_encoder',
    id: getPrefixedId('pos_prompt'),
    use_prompt_enhancer: usePromptEnhancer,
  });

  let negCond: Invocation<'ernie_image_text_encoder'> | null = null;
  if (guidance_scale > 1) {
    negCond = g.addNode({
      type: 'ernie_image_text_encoder',
      id: getPrefixedId('neg_prompt'),
      prompt: prompts.negative ?? '',
      // Negative prompt should not be PE-enhanced.
      use_prompt_enhancer: false,
    });
  }

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });

  const denoise = g.addNode({
    type: 'ernie_image_denoise',
    id: getPrefixedId('denoise_latents'),
    guidance_scale,
    steps,
    scheduler: ernieImageScheduler,
  });

  const l2i = g.addNode({
    type: 'ernie_image_vae_decode',
    id: getPrefixedId('l2i'),
  });

  // Wire transformer / VAE / text encoder
  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'text_encoder', posCond, 'text_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  // Optional prompt-enhancer wiring (only if the loader emits one and the toggle is on)
  if (usePromptEnhancer) {
    g.addEdge(modelLoader, 'prompt_enhancer', posCond, 'prompt_enhancer');
  }

  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(posCond, 'conditioning', denoise, 'positive_conditioning');

  if (negCond !== null) {
    g.addEdge(modelLoader, 'text_encoder', negCond, 'text_encoder');
    g.addEdge(negCond, 'conditioning', denoise, 'negative_conditioning');
  }

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'ernie-image');

  g.upsertMetadata({
    cfg_scale: guidance_scale,
    negative_prompt: prompts.negative,
    model: Graph.getModelMetadataField(modelConfig),
    steps,
    scheduler: ernieImageScheduler,
    ernie_image_use_prompt_enhancer: usePromptEnhancer,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage({ g, state, denoise, l2i });
    g.upsertMetadata({ generation_mode: 'ernie_image_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'ernie_image_vae_encode',
      id: getPrefixedId('ernie_image_i2l'),
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
    g.upsertMetadata({ generation_mode: 'ernie_image_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'ernie_image_vae_encode',
      id: getPrefixedId('ernie_image_i2l'),
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
    g.upsertMetadata({ generation_mode: 'ernie_image_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'ernie_image_vae_encode',
      id: getPrefixedId('ernie_image_i2l'),
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
    g.upsertMetadata({ generation_mode: 'ernie_image_outpaint' });
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
