import { objectEquals } from '@observ33r/object-equals';
import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
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
import { collectIdeogram4PromptInputs } from 'features/nodes/util/graph/generation/buildIdeogram4Prompt';
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
  // Optional advanced overrides (null = use the preset).
  const ideogram4Steps = selectIdeogram4Steps(state);
  const ideogram4GuidanceScale = selectIdeogram4GuidanceScale(state);
  const ideogram4Mu = selectIdeogram4Mu(state);

  // Collect the raw prompt inputs. The JSON caption is assembled at generation time in the
  // ideogram4_caption_builder node so dynamic prompts / prompt batching (which vary the global
  // prompt) are folded into the encoded caption. The regions and palette are fixed per generation.
  const { globalPrompt, regions, colorPalette } = collectIdeogram4PromptInputs(state, manager);
  // Whether anything beyond the global prompt shapes the caption. When true we record the assembled
  // caption in metadata (via an edge off the runtime builder) so it captures each batch item's caption.
  const hasStructuredInputs = regions.length > 0 || colorPalette.length > 0;

  const g = new Graph(getPrefixedId('ideogram4_graph'));

  const modelLoader = g.addNode({
    type: 'ideogram4_model_loader',
    id: getPrefixedId('ideogram4_model_loader'),
    model,
  });

  // The global prompt lives in its own node so the linear batch (dynamic prompts / prompt batching)
  // can inject expansions into it — those then flow through the caption builder into the encoder.
  const promptNode = g.addNode({
    id: getPrefixedId('ideogram4_prompt'),
    type: 'string',
    value: globalPrompt,
  });

  // Assembles the structured JSON caption from the (possibly batch-injected) prompt + fixed regions.
  const captionBuilder = g.addNode({
    id: getPrefixedId('ideogram4_caption_builder'),
    type: 'ideogram4_caption_builder',
    regions,
    color_palette: colorPalette,
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
  g.addEdge(promptNode, 'value', captionBuilder, 'prompt');
  g.addEdge(captionBuilder, 'value', textEncoder, 'prompt');
  g.addEdge(textEncoder, 'conditioning', denoise, 'positive_conditioning');
  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  // Text-to-image dimensions. Ideogram 4 requires multiples of 16 (enforced by the bbox grid size).
  const { originalSize, scaledSize } = getOriginalAndScaledSizesForTextToImage(state);
  denoise.width = scaledSize.width;
  denoise.height = scaledSize.height;

  // Return the real prompt node so the linear batch injects dynamic-prompt expansions into it; they
  // flow through the caption builder into the encoder. `positive_prompt` metadata is the global prompt.
  const positivePrompt: Invocation<'string'> = promptNode;

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
  // Record the actually-encoded caption for reproducibility. It's assembled at runtime, so take it via
  // an edge off the builder — this captures each batch item's caption, not a stale build-time value.
  // Only when something beyond the plain global prompt shapes it (otherwise it equals positive_prompt).
  if (hasStructuredInputs) {
    g.addEdgeToMetadata(captionBuilder, 'value', 'ideogram4_caption');
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
