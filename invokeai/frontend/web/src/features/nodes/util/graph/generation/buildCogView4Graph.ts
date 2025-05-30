import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasMetadata, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { getGenerationMode } from 'features/nodes/util/graph/generation/getGenerationMode';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getSizes,
  selectCanvasOutputFields,
  selectPresetModifiedPrompts,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

export const buildCogView4Graph = async (
  state: RootState,
  manager?: CanvasManager | null
): Promise<GraphBuilderReturn> => {
  const generationMode = await getGenerationMode(manager);
  log.debug({ generationMode }, 'Building CogView4 graph');

  const params = selectParamsSlice(state);
  const canvas = selectCanvasSlice(state);

  const { bbox } = canvas;

  const { model, cfgScale: cfg_scale, seed, steps } = params;

  assert(model, 'No model found in state');

  const { originalSize, scaledSize } = getSizes(bbox);
  const { positivePrompt, negativePrompt } = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('cogview4_graph'));
  const modelLoader = g.addNode({
    type: 'cogview4_model_loader',
    id: getPrefixedId('cogview4_model_loader'),
    model,
  });
  const posCond = g.addNode({
    type: 'cogview4_text_encoder',
    id: getPrefixedId('pos_prompt'),
    prompt: positivePrompt,
  });

  const negCond = g.addNode({
    type: 'cogview4_text_encoder',
    id: getPrefixedId('neg_prompt'),
    prompt: negativePrompt,
  });

  const denoise = g.addNode({
    type: 'cogview4_denoise',
    id: getPrefixedId('denoise_latents'),
    cfg_scale,
    width: scaledSize.width,
    height: scaledSize.height,
    steps,
    denoising_start: 0,
    denoising_end: 1,
  });
  const l2i = g.addNode({
    type: 'cogview4_l2i',
    id: getPrefixedId('l2i'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'glm_encoder', posCond, 'glm_encoder');
  g.addEdge(modelLoader, 'glm_encoder', negCond, 'glm_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  g.addEdge(posCond, 'conditioning', denoise, 'positive_conditioning');
  g.addEdge(negCond, 'conditioning', denoise, 'negative_conditioning');

  g.addEdge(denoise, 'latents', l2i, 'latents');

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'cogview4');

  g.upsertMetadata({
    cfg_scale,
    width: originalSize.width,
    height: originalSize.height,
    positive_prompt: positivePrompt,
    negative_prompt: negativePrompt,
    model: Graph.getModelMetadataField(modelConfig),
    seed,
    steps,
  });

  const denoising_start = 1 - params.img2imgStrength;

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage({ g, l2i, originalSize, scaledSize });
    g.upsertMetadata({ generation_mode: 'cogview4_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager, 'Need manager to do img2img');
    canvasOutput = await addImageToImage({
      g,
      manager,
      l2i,
      i2lNodeType: 'cogview4_i2l',
      denoise,
      vaeSource: modelLoader,
      originalSize,
      scaledSize,
      bbox,
      denoising_start,
      fp32: false,
    });
    g.upsertMetadata({ generation_mode: 'cogview4_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager, 'Need manager to do inpaint');
    canvasOutput = await addInpaint({
      state,
      g,
      manager,
      l2i,
      i2lNodeType: 'cogview4_i2l',
      denoise,
      vaeSource: modelLoader,
      modelLoader,
      originalSize,
      scaledSize,
      denoising_start,
      fp32: false,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'cogview4_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager, 'Need manager to do outpaint');
    canvasOutput = await addOutpaint({
      state,
      g,
      manager,
      l2i,
      i2lNodeType: 'cogview4_i2l',
      denoise,
      vaeSource: modelLoader,
      modelLoader,
      originalSize,
      scaledSize,
      denoising_start,
      fp32: false,
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

  g.upsertMetadata(selectCanvasMetadata(state));

  g.updateNode(canvasOutput, selectCanvasOutputFields(state));

  g.setMetadataReceivingNode(canvasOutput);
  return {
    g,
    seedFieldIdentifier: { nodeId: denoise.id, fieldName: 'seed' },
    positivePromptFieldIdentifier: { nodeId: posCond.id, fieldName: 'prompt' },
  };
};
