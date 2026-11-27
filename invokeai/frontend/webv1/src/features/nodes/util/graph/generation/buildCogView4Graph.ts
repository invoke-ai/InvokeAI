import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
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

export const buildCogView4Graph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building CogView4 graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'cogview4', 'Selected model is not a CogView4 model');

  const params = selectParamsSlice(state);

  const { cfgScale: cfg_scale, steps } = params;

  const prompts = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('cogview4_graph'));

  const modelLoader = g.addNode({
    type: 'cogview4_model_loader',
    id: getPrefixedId('cogview4_model_loader'),
    model,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'cogview4_text_encoder',
    id: getPrefixedId('pos_prompt'),
  });

  const negCond = g.addNode({
    type: 'cogview4_text_encoder',
    id: getPrefixedId('neg_prompt'),
    prompt: prompts.negative,
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  const denoise = g.addNode({
    type: 'cogview4_denoise',
    id: getPrefixedId('denoise_latents'),
    cfg_scale,
    steps,
  });
  const l2i = g.addNode({
    type: 'cogview4_l2i',
    id: getPrefixedId('l2i'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'glm_encoder', posCond, 'glm_encoder');
  g.addEdge(modelLoader, 'glm_encoder', negCond, 'glm_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(posCond, 'conditioning', denoise, 'positive_conditioning');

  g.addEdge(negCond, 'conditioning', denoise, 'negative_conditioning');

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'cogview4');

  g.upsertMetadata({
    cfg_scale,
    negative_prompt: prompts.negative,
    model: Graph.getModelMetadataField(modelConfig),
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
    g.upsertMetadata({ generation_mode: 'cogview4_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'cogview4_i2l',
      id: getPrefixedId('cogview4_i2l'),
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
    g.upsertMetadata({ generation_mode: 'cogview4_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'cogview4_i2l',
      id: getPrefixedId('cogview4_i2l'),
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
    g.upsertMetadata({ generation_mode: 'cogview4_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'cogview4_i2l',
      id: getPrefixedId('cogview4_i2l'),
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
    g.upsertMetadata({ generation_mode: 'cogview4_outpaint' });
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
