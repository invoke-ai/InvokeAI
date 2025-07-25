import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasMetadata, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { t } from 'i18next';
import type { Invocation } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

import { addControlNets } from './addControlAdapters';

const log = logger('system');

export const buildBriaGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;
  log.debug({ generationMode, manager: manager?.id }, 'Building Bria graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'bria', 'Selected model is not a Bria model');

  const params = selectParamsSlice(state);
  const canvas = selectCanvasSlice(state);
  const refImages = selectRefImagesSlice(state);

  const { guidance, steps, seed } = params;

  // Bria only supports txt2img for now
  if (generationMode !== 'txt2img') {
    throw new UnsupportedGenerationModeError(t('toast.briaIncompatibleGenerationMode'));
  }

  const g = new Graph(getPrefixedId('bria_graph'));

  // Add model loader
  const modelLoader = g.addNode({
    type: 'bria_model_loader',
    id: getPrefixedId('bria_model_loader'),
    model,
  } as Invocation<'bria_model_loader'>);

  // Add positive prompt
  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });

  // Add text encoder
  const textEncoder = g.addNode({
    type: 'bria_text_encoder',
    id: getPrefixedId('bria_text_encoder'),
    prompt: positivePrompt,
    negative_prompt: params.negativePrompt,
    max_length: 128,
  } as Invocation<'bria_text_encoder'>);

  // Add latent sampler for initial noise
  const latentSampler = g.addNode({
    type: 'bria_latent_sampler',
    id: getPrefixedId('bria_latent_sampler'),
    width: params.width,
    height: params.height,
    seed: seed,
  } as Invocation<'bria_latent_sampler'>);

  // Add denoise node
  const denoise = g.addNode({
    type: 'bria_denoise',
    id: getPrefixedId('bria_denoise'),
    num_steps: steps,
    guidance_scale: guidance,
  } as Invocation<'bria_denoise'>);

  // Add decoder
  const decoder = g.addNode({
    type: 'bria_decoder',
    id: getPrefixedId('bria_decoder'),
  } as Invocation<'bria_decoder'>);

  // Connect model components to text encoder
  g.addEdge(modelLoader, 't5_encoder', textEncoder, 't5_encoder');

  // Connect model components to latent sampler
  g.addEdge(modelLoader, 'transformer', latentSampler, 'transformer');

  // Connect model components to denoise
  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 't5_encoder', denoise, 't5_encoder');
  g.addEdge(modelLoader, 'vae', denoise, 'vae');

  // Connect text encoder to denoise
  g.addEdge(textEncoder, 'pos_embeds', denoise, 'pos_embeds');
  g.addEdge(textEncoder, 'neg_embeds', denoise, 'neg_embeds');
  g.addEdge(textEncoder, 'text_ids', denoise, 'text_ids');

  // Connect latent sampler to denoise
  g.addEdge(latentSampler, 'latents', denoise, 'latents');
  g.addEdge(latentSampler, 'latent_image_ids', denoise, 'latent_image_ids');

  // Connect model components to decoder
  g.addEdge(modelLoader, 'vae', decoder, 'vae');

  // Connect denoise to decoder
  g.addEdge(denoise, 'latents', decoder, 'latents');

  // Add ControlNet support
  if (manager !== null) {
    const controlNetCollector = g.addNode({
      type: 'collect',
      id: getPrefixedId('control_net_collector'),
    });
    
    const controlNetResult = await addControlNets({
      manager,
      entities: canvas.controlLayers.entities,
      g,
      rect: canvas.bbox.rect,
      collector: controlNetCollector,
      model,
    });
    
    if (controlNetResult.addedControlNets > 0) {
      // Connect the collector to the denoise node's control input
      g.addEdge(controlNetCollector, 'collection', denoise, 'control');
    } else {
      // Remove the collector if no control nets were added
      g.deleteNode(controlNetCollector.id);
    }
  }

  // Add metadata
  g.upsertMetadata({
    guidance_scale: guidance,
    model: Graph.getModelMetadataField(model),
    steps,
    generation_mode: 'bria_txt2img',
  });
  g.addEdgeToMetadata(latentSampler, 'seed', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  let canvasOutput: Invocation<ImageOutputNodes> = decoder;

  // Add text to image handling
  canvasOutput = addTextToImage({
    g,
    state,
    denoise: decoder, // Use decoder as the denoise equivalent
    l2i: decoder,
  });

  // Add NSFW checker
  if (state.system.shouldUseNSFWChecker) {
    canvasOutput = addNSFWChecker(g, canvasOutput);
  }

  // Add watermarker
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
    seed: latentSampler,
    positivePrompt,
  };
}; 