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
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getOriginalAndScaledSizesForTextToImage,
  selectCanvasOutputFields,
  selectPresetModifiedPrompts,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
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
  });

  let negCond: Invocation<'ernie_image_text_encoder'> | null = null;
  if (guidance_scale > 1) {
    negCond = g.addNode({
      type: 'ernie_image_text_encoder',
      id: getPrefixedId('neg_prompt'),
      prompt: prompts.negative ?? '',
    });
  }

  // The enhancer is its own node so that it stays on the session's GPU: it samples an
  // autoregressive rewrite on every generation, which is far too long to hold a borrowed idle GPU
  // for, whereas the encoders it feeds are `idle_gpu_offloadable`. Only the positive prompt goes
  // through it -- the negative encoder takes its prompt as a literal field, unenhanced.
  let promptEnhancer: Invocation<'ernie_image_prompt_enhancer'> | null = null;
  if (usePromptEnhancer) {
    promptEnhancer = g.addNode({
      type: 'ernie_image_prompt_enhancer',
      id: getPrefixedId('prompt_enhancer'),
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

  // Optional prompt-enhancer wiring (only if the loader emits one and the toggle is on). The
  // enhancer sits between the prompt node and the encoder; without it the prompt goes straight in.
  if (promptEnhancer !== null) {
    g.addEdge(modelLoader, 'prompt_enhancer', promptEnhancer, 'prompt_enhancer');
    g.addEdge(positivePrompt, 'value', promptEnhancer, 'prompt');
    g.addEdge(promptEnhancer, 'value', posCond, 'prompt');
  } else {
    g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  }

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

  // ERNIE-Image is text-to-image only. Its denoise node has no `denoise_mask` input, so
  // masked modes (inpaint/outpaint) are unsupported, and we do not offer image-to-image.
  if (generationMode !== 'txt2img') {
    throw new UnsupportedGenerationModeError(
      `ERNIE-Image only supports text-to-image generation, but got generation mode: ${generationMode}`
    );
  }

  let canvasOutput: Invocation<ImageOutputNodes> = addTextToImage({ g, state, denoise, l2i });
  g.upsertMetadata({ generation_mode: 'ernie_image_txt2img' });

  // The prompt enhancer is handed the target size and rewrites the prompt to suit that aspect
  // ratio, so it needs the real dimensions rather than the node's 1024x1024 defaults. Use the
  // *original* size: with canvas scaling active the denoise node carries the intermediate render
  // size, but what the user ends up looking at is the original.
  if (promptEnhancer !== null) {
    const { originalSize } = getOriginalAndScaledSizesForTextToImage(state);
    promptEnhancer.width = originalSize.width;
    promptEnhancer.height = originalSize.height;
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
