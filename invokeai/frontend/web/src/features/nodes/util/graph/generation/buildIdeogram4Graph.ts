import { objectEquals } from '@observ33r/object-equals';
import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  selectIdeogram4ColorPalette,
  selectIdeogram4GuidanceScale,
  selectIdeogram4Mu,
  selectIdeogram4SamplerPreset,
  selectIdeogram4Steps,
  selectMainModelConfig,
} from 'features/controlLayers/store/paramsSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { buildIdeogram4Prompt } from 'features/nodes/util/graph/generation/buildIdeogram4Prompt';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getOriginalAndScaledSizesForTextToImage,
  selectCanvasOutputFields,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('system');

/**
 * Builds the graph for Ideogram 4 generation. Ideogram 4 is text-to-image only and prompted with a
 * structured JSON caption (assembled from the global prompt + Canvas Regional Guidance layers; see
 * buildIdeogram4Prompt). There is no negative prompt (the reference uses an asymmetric CFG with a
 * zeroed unconditional branch), and no img2img/inpaint/outpaint or mask conditioning.
 */
export const buildIdeogram4Graph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Ideogram 4 graph');

  assert(generationMode === 'txt2img', 'Ideogram 4 only supports text-to-image generation');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'ideogram-4', 'Selected model is not an Ideogram 4 model');

  const samplerPreset = selectIdeogram4SamplerPreset(state);
  // Optional advanced overrides (null = use the preset). The color palette is consumed by
  // buildIdeogram4Prompt; it is read here only for metadata.
  const ideogram4Steps = selectIdeogram4Steps(state);
  const ideogram4GuidanceScale = selectIdeogram4GuidanceScale(state);
  const ideogram4Mu = selectIdeogram4Mu(state);
  const colorPalette = selectIdeogram4ColorPalette(state);

  // Assemble the prompt: raw-JSON passthrough, a structured caption built from Regional Guidance
  // layers (+ optional color palette), or plain text when there is nothing structured to encode.
  const { prompt, isStructured } = buildIdeogram4Prompt(state, manager);

  const g = new Graph(getPrefixedId('ideogram4_graph'));

  const modelLoader = g.addNode({
    type: 'ideogram4_model_loader',
    id: getPrefixedId('ideogram4_model_loader'),
    model,
  });

  const promptNode = g.addNode({
    id: getPrefixedId('ideogram4_prompt'),
    type: 'string',
    value: prompt,
  });

  const textEncoder = g.addNode({
    type: 'ideogram4_text_encoder',
    id: getPrefixedId('ideogram4_text_encoder'),
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });

  const denoise = g.addNode({
    type: 'ideogram4_denoise',
    id: getPrefixedId('ideogram4_denoise'),
    sampler_preset: samplerPreset,
    steps: ideogram4Steps ?? undefined,
    guidance_scale: ideogram4GuidanceScale ?? undefined,
    mu: ideogram4Mu ?? undefined,
  });

  const l2i = g.addNode({
    type: 'ideogram4_l2i',
    id: getPrefixedId('ideogram4_l2i'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'qwen3_encoder', textEncoder, 'qwen3_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');
  g.addEdge(promptNode, 'value', textEncoder, 'prompt');
  g.addEdge(textEncoder, 'conditioning', denoise, 'positive_conditioning');
  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  // Text-to-image dimensions. Ideogram 4 requires multiples of 16 (enforced by the bbox grid size).
  const { originalSize, scaledSize } = getOriginalAndScaledSizesForTextToImage(state);
  denoise.width = scaledSize.width;
  denoise.height = scaledSize.height;

  // The linear batch injects the raw positive prompt (and dynamic-prompt expansions) into the node we
  // return as `positivePrompt`. For a structured caption we must NOT let it clobber the assembled JSON,
  // so we return a decoy string node; for plain text we return the real prompt node so dynamic prompts
  // and prompt batching work normally.
  const positivePrompt: Invocation<'string'> = isStructured
    ? g.addNode({ id: getPrefixedId('positive_prompt_decoy'), type: 'string' })
    : promptNode;

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'ideogram-4');

  g.upsertMetadata({
    model: Graph.getModelMetadataField(modelConfig),
    ideogram4_sampler_preset: samplerPreset,
    // 'auto' sentinel survives the backend's exclude_none metadata serialization; the parser maps it
    // back to null (use preset) on recall.
    ideogram4_steps: ideogram4Steps ?? 'auto',
    ideogram4_guidance_scale: ideogram4GuidanceScale ?? 'auto',
    ideogram4_mu: ideogram4Mu ?? 'auto',
    ideogram4_color_palette: colorPalette,
    width: originalSize.width,
    height: originalSize.height,
    generation_mode: 'ideogram4_txt2img',
  });
  // The assembled caption is static; store it for reproducibility when it differs from the raw prompt.
  if (isStructured) {
    g.upsertMetadata({ ideogram4_caption: prompt });
  }
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  // Resize the output back to the original size if the canvas used a scaled bbox.
  let canvasOutput: Invocation<ImageOutputNodes> = l2i;
  if (!objectEquals(scaledSize, originalSize)) {
    const resizeImageToOriginalSize = g.addNode({
      id: getPrefixedId('resize_image_to_original_size'),
      type: 'img_resize',
      ...originalSize,
    });
    g.addEdge(l2i, 'image', resizeImageToOriginalSize, 'image');
    canvasOutput = resizeImageToOriginalSize;
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
