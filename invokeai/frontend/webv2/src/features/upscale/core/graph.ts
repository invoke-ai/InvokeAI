import type {
  BackendGraphContract,
  BackendInvocationContract,
  GenerationProjectSettings,
  MainModelConfig,
  ResultDestination,
} from '@features/generation/contracts';

import {
  addLoraCollectionLoader,
  addEdge,
  addNode,
  addTransformerLoraCollectionLoader,
  getActiveCompatibleLoras,
  toGraphContract,
  toModelIdentifier,
} from '@features/generation/graph';
import { coerceSchedulerForGraph } from '@features/generation/settings';

import type { UpscaleBase } from './settings';
import type { CompiledUpscaleGraph, UpscaleWidgetValues } from './types';

import { getUpscaleValidationReasons, upscaleArchitectureFor, upscaleDenoiseDimension } from './settings';

export const getUpscaleDenoisingStart = (creativity: number): number => ((creativity * -1 + 10) * 4.99) / 100;

export const getUpscaleControlNetValues = (structure: number) => {
  const splitPoint = (structure + 10) * 0.025 + 0.3;

  return {
    first: {
      beginStepPercent: 0,
      controlWeight: (structure + 10) * 0.0325 + 0.3,
      endStepPercent: splitPoint,
    },
    second: {
      beginStepPercent: splitPoint,
      controlWeight: ((structure + 10) * 0.0325 + 0.15) * 0.45,
      endStepPercent: 0.85,
    },
  };
};

const addUpscaleMetadata = (
  graph: BackendGraphContract,
  output: BackendInvocationContract,
  settings: UpscaleWidgetValues,
  projectSettings: GenerationProjectSettings,
  randDevice: string
): void => {
  if (!settings.model || !settings.upscaleModel || !settings.inputImage) {
    return;
  }

  // Pass only loras; fabricating unrelated GenerateSettings fields would obscure the helper's actual dependency.
  const activeLoras = getActiveCompatibleLoras({ loras: settings.loras }, settings.model);
  const architecture = upscaleArchitectureFor(settings.model);
  // FLUX calls this guidance, not CFG, and its encoders have to be recalled with the rest.
  const guidanceField = architecture?.usesNegativePrompt
    ? { cfg_scale: settings.cfgScale }
    : { guidance: settings.cfgScale };
  const metadata = addNode(graph, {
    ...guidanceField,
    ...(settings.t5EncoderModel ? { t5_encoder: toModelIdentifier(settings.t5EncoderModel) } : {}),
    ...(settings.clipEmbedModel ? { clip_embed_model: toModelIdentifier(settings.clipEmbedModel) } : {}),
    creativity: settings.creativity,
    id: 'core_metadata',
    loras: activeLoras.map((lora) => ({ model: toModelIdentifier(lora.model), weight: lora.weight })),
    model: toModelIdentifier(settings.model),
    rand_device: randDevice,
    scheduler: coerceSchedulerForGraph(settings.model, settings.scheduler),
    steps: settings.steps,
    structure: settings.structure,
    tile_overlap: settings.tileOverlap,
    tile_size: settings.tileSize,
    type: 'core_metadata',
    upscale_initial_image: {
      height: settings.inputImage.height,
      image_name: settings.inputImage.image_name,
      width: settings.inputImage.width,
    },
    upscale_model: toModelIdentifier(settings.upscaleModel),
    upscale_scale: settings.scale,
    vae: settings.vae ? toModelIdentifier(settings.vae) : undefined,
  });

  addEdge(graph, graph.nodes.positive_prompt, 'value', metadata, 'positive_prompt');
  if (architecture?.usesNegativePrompt) {
    addEdge(graph, graph.nodes.negative_prompt, 'value', metadata, 'negative_prompt');
  }
  addEdge(graph, graph.nodes.seed, 'value', metadata, 'seed');
  addEdge(graph, graph.nodes.spandrel_autoscale, 'width', metadata, 'width');
  addEdge(graph, graph.nodes.spandrel_autoscale, 'height', metadata, 'height');
  addEdge(graph, metadata, 'metadata', output, 'metadata');
};

/**
 * Everything the second pass needs, once the frame has been enlarged.
 *
 * The prefix — prompts, seed, Spandrel, unsharp — is identical for every architecture, so it is
 * built once and handed over. What differs is the model loader, the conditioning, the VAE ends and
 * whether anything anchors the pass to the enlarged frame.
 */
interface UpscalePathContext {
  graph: BackendGraphContract;
  settings: UpscaleWidgetValues;
  model: MainModelConfig;
  destination: ResultDestination;
  projectSettings: GenerationProjectSettings;
  positivePrompt: BackendInvocationContract;
  negativePrompt: BackendInvocationContract;
  seed: BackendInvocationContract;
  unsharpMask: BackendInvocationContract;
  /** The frame size the second pass will run at, already floored to the architecture's grid. */
  denoiseSize: { width: number; height: number };
}

interface UpscalePathResult {
  /** The node whose image is the result; the caller hangs metadata off it. */
  output: BackendInvocationContract;
  /** Names the pass in the queue, so "multi-diffusion" is not claimed where none runs. */
  title: string;
}

type UpscalePathBuilder = (context: UpscalePathContext) => UpscalePathResult;

const buildSdUpscalePath: UpscalePathBuilder = ({
  graph,
  settings,
  model,
  destination,
  projectSettings,
  positivePrompt,
  negativePrompt,
  seed,
  unsharpMask,
}) => {
  const { tileControlnetModel } = settings;

  if (!tileControlnetModel) {
    throw new Error('Upscale needs a Tile or Union ControlNet compatible with the main model.');
  }

  const noise = addNode(graph, { id: 'noise', type: 'noise', use_cpu: projectSettings.useCpuNoise });
  const imageToLatents = addNode(graph, {
    fp32: settings.vaePrecision === 'fp32',
    id: 'i2l',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'i2l',
  });
  const output = addNode(graph, {
    fp32: settings.vaePrecision === 'fp32',
    id: 'upscale_output',
    is_intermediate: destination === 'canvas',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'l2i',
    use_cache: false,
  });
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    denoising_end: 1,
    denoising_start: getUpscaleDenoisingStart(settings.creativity),
    id: 'tiled_multidiffusion_denoise_latents',
    scheduler: coerceSchedulerForGraph(model, settings.scheduler),
    steps: settings.steps,
    tile_height: settings.tileSize,
    tile_overlap: settings.tileOverlap,
    tile_width: settings.tileSize,
    type: 'tiled_multi_diffusion_denoise_latents',
  });
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model: toModelIdentifier(model),
    type: model.base === 'sdxl' ? 'sdxl_model_loader' : 'main_model_loader',
  });
  const compelType = model.base === 'sdxl' ? 'sdxl_compel_prompt' : 'compel';
  const posCond = addNode(graph, { id: 'pos_cond', type: compelType });
  const negCond = addNode(graph, { id: 'neg_cond', type: compelType });
  const activeLoras = settings.loras.filter(
    (lora) =>
      lora.isEnabled && (lora.model.base === model.base || (model.base === 'sdxl' && lora.model.base === 'sdxl'))
  );
  let unetSource: BackendInvocationContract = modelLoader;
  let clipSource: BackendInvocationContract = modelLoader;
  let clip2Source: BackendInvocationContract | undefined = model.base === 'sdxl' ? modelLoader : undefined;

  addEdge(graph, seed, 'value', noise, 'seed');
  addEdge(graph, unsharpMask, 'width', noise, 'width');
  addEdge(graph, unsharpMask, 'height', noise, 'height');
  addEdge(graph, unsharpMask, 'image', imageToLatents, 'image');

  if (model.base === 'sdxl') {
    addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
    addEdge(graph, positivePrompt, 'value', posCond, 'style');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
    addEdge(graph, negativePrompt, 'value', negCond, 'style');
  } else {
    const clipSkip = addNode(graph, { id: 'clip_skip', skipped_layers: settings.clipSkip, type: 'clip_skip' });

    addEdge(graph, modelLoader, 'clip', clipSkip, 'clip');
    clipSource = clipSkip;
    addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
    addEdge(graph, negativePrompt, 'value', negCond, 'prompt');
  }

  if (activeLoras.length > 0) {
    const loraLoader = addLoraCollectionLoader(graph, activeLoras, model, {
      clip: clipSource,
      clip2: clip2Source,
      unet: unetSource,
    });

    unetSource = loraLoader;
    clipSource = loraLoader;
    clip2Source = model.base === 'sdxl' ? loraLoader : undefined;
  }

  addEdge(graph, unetSource, 'unet', denoise, 'unet');
  addEdge(graph, clipSource, 'clip', posCond, 'clip');
  addEdge(graph, clipSource, 'clip', negCond, 'clip');
  if (model.base === 'sdxl') {
    addEdge(graph, clip2Source ?? modelLoader, 'clip2', posCond, 'clip2');
    addEdge(graph, clip2Source ?? modelLoader, 'clip2', negCond, 'clip2');
  }

  const vaeLoader =
    settings.vae && settings.vae.base === model.base
      ? addNode(graph, { id: 'vae_loader', type: 'vae_loader', vae_model: toModelIdentifier(settings.vae) })
      : null;
  const vaeSource = vaeLoader ?? modelLoader;

  addEdge(graph, vaeSource, 'vae', imageToLatents, 'vae');
  addEdge(graph, vaeSource, 'vae', output, 'vae');
  addEdge(graph, noise, 'noise', denoise, 'noise');
  addEdge(graph, imageToLatents, 'latents', denoise, 'latents');
  addEdge(graph, posCond, 'conditioning', denoise, 'positive_conditioning');
  addEdge(graph, negCond, 'conditioning', denoise, 'negative_conditioning');
  addEdge(graph, denoise, 'latents', output, 'latents');

  const control = getUpscaleControlNetValues(settings.structure);
  const controlNet1 = addNode(graph, {
    begin_step_percent: control.first.beginStepPercent,
    control_mode: 'balanced',
    control_model: toModelIdentifier(tileControlnetModel),
    control_weight: control.first.controlWeight,
    end_step_percent: control.first.endStepPercent,
    id: 'controlnet_1',
    resize_mode: 'just_resize',
    type: 'controlnet',
  });
  const controlNet2 = addNode(graph, {
    begin_step_percent: control.second.beginStepPercent,
    control_mode: 'balanced',
    control_model: toModelIdentifier(tileControlnetModel),
    control_weight: control.second.controlWeight,
    end_step_percent: control.second.endStepPercent,
    id: 'controlnet_2',
    resize_mode: 'just_resize',
    type: 'controlnet',
  });
  const controlNetCollector = addNode(graph, { id: 'controlnet_collector', type: 'collect' });

  addEdge(graph, unsharpMask, 'image', controlNet1, 'image');
  addEdge(graph, unsharpMask, 'image', controlNet2, 'image');
  addEdge(graph, controlNet1, 'control', controlNetCollector, 'item');
  addEdge(graph, controlNet2, 'control', controlNetCollector, 'item');
  addEdge(graph, controlNetCollector, 'collection', denoise, 'control');

  return { output, title: `${model.name} multi-diffusion upscale` };
};

/**
 * FLUX.1 has no tile ControlNet, so the second pass is anchored only by where it starts on the
 * schedule. Both VAE ends tile: the frame handed to `i2l` is the largest in the graph, and an
 * untiled encode of it does not fit on consumer hardware.
 */
const buildFluxUpscalePath: UpscalePathBuilder = ({
  graph,
  settings,
  model,
  destination,
  positivePrompt,
  seed,
  unsharpMask,
  denoiseSize,
}) => {
  const modelLoader = addNode(graph, {
    clip_embed_model: settings.clipEmbedModel ? toModelIdentifier(settings.clipEmbedModel) : undefined,
    id: 'model_loader',
    model: toModelIdentifier(model),
    t5_encoder_model: settings.t5EncoderModel ? toModelIdentifier(settings.t5EncoderModel) : undefined,
    type: 'flux_model_loader',
    vae_model: settings.vae ? toModelIdentifier(settings.vae) : undefined,
  });
  const activeLoras = getActiveCompatibleLoras({ loras: settings.loras }, model);
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'flux_lora_collection_loader', modelLoader, [
        'transformer',
        'clip',
        't5_encoder',
      ])
    : modelLoader;
  const posCond = addNode(graph, { id: 'pos_cond', type: 'flux_text_encoder' });
  const posCondCollect = addNode(graph, { id: 'pos_cond_collect', type: 'collect' });
  // Spandrel rounds to a multiple of 8; `flux_denoise` accepts only multiples of 16 and rejects the
  // rest *after* the upscale has run. Resizing here makes the frame, its latents and the denoise
  // size agree, and lets the compiler state the size rather than defer it to an edge.
  const fitToGrid = addNode(graph, {
    height: denoiseSize.height,
    id: 'fit_to_grid',
    resample_mode: 'bicubic',
    type: 'img_resize',
    width: denoiseSize.width,
  });
  const imageToLatents = addNode(graph, {
    id: 'i2l',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'flux_vae_encode',
  });
  const denoise = addNode(graph, {
    denoising_end: 1,
    denoising_start: getUpscaleDenoisingStart(settings.creativity),
    guidance: settings.cfgScale,
    height: denoiseSize.height,
    id: 'denoise_latents',
    num_steps: settings.steps,
    scheduler: coerceSchedulerForGraph(model, settings.scheduler),
    type: 'flux_denoise',
    width: denoiseSize.width,
  });
  const output = addNode(graph, {
    id: 'upscale_output',
    is_intermediate: destination === 'canvas',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'flux_vae_decode',
    use_cache: false,
  });

  addEdge(graph, loraSource, 'clip', posCond, 'clip');
  addEdge(graph, loraSource, 't5_encoder', posCond, 't5_encoder');
  addEdge(graph, modelLoader, 'max_seq_len', posCond, 't5_max_seq_len');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_text_conditioning');
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, seed, 'value', denoise, 'seed');

  addEdge(graph, unsharpMask, 'image', fitToGrid, 'image');
  addEdge(graph, fitToGrid, 'image', imageToLatents, 'image');
  addEdge(graph, modelLoader, 'vae', imageToLatents, 'vae');
  addEdge(graph, imageToLatents, 'latents', denoise, 'latents');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addEdge(graph, modelLoader, 'vae', output, 'vae');

  return { output, title: `${model.name} upscale` };
};

/**
 * Keyed by the same bases as `UPSCALE_ARCHITECTURES`, and `satisfies` makes that a compile error to
 * break: declaring an architecture supported without giving it a builder will not type-check.
 */
const UPSCALE_PATH_BUILDERS = {
  flux: buildFluxUpscalePath,
  'sd-1': buildSdUpscalePath,
  sdxl: buildSdUpscalePath,
} satisfies Record<UpscaleBase, UpscalePathBuilder>;

export const compileUpscaleGraph = (
  settings: UpscaleWidgetValues,
  destination: ResultDestination,
  projectSettings: GenerationProjectSettings,
  randDevice = projectSettings.useCpuNoise ? 'cpu' : 'cuda'
): CompiledUpscaleGraph => {
  const reasons = getUpscaleValidationReasons(settings);

  if (reasons.length > 0) {
    throw new Error(reasons[0]);
  }

  const { inputImage, model, upscaleModel } = settings;

  if (!inputImage || !model || !upscaleModel) {
    throw new Error('Upscale settings are incomplete.');
  }

  const buildPath = UPSCALE_PATH_BUILDERS[model.base as UpscaleBase];

  if (!buildPath) {
    throw new Error(`Upscale does not support ${model.base} main models.`);
  }

  const grid = upscaleArchitectureFor(model)?.denoiseGrid ?? 8;
  const denoiseSize = {
    height: upscaleDenoiseDimension(inputImage.height, settings.scale, grid),
    width: upscaleDenoiseDimension(inputImage.width, settings.scale, grid),
  };
  const graph: BackendGraphContract = { edges: [], id: 'upscale-graph', nodes: {} };
  const positivePrompt = addNode(graph, { id: 'positive_prompt', type: 'string' });
  const negativePrompt = addNode(graph, { id: 'negative_prompt', type: 'string' });
  const seed = addNode(graph, { id: 'seed', type: 'integer' });
  const spandrelAutoscale = addNode(graph, {
    fit_to_multiple_of_8: true,
    id: 'spandrel_autoscale',
    image: { image_name: inputImage.image_name },
    image_to_image_model: toModelIdentifier(upscaleModel),
    scale: settings.scale,
    type: 'spandrel_image_to_image_autoscale',
  });
  const unsharpMask = addNode(graph, { id: 'unsharp_2', radius: 2, strength: 60, type: 'unsharp_mask' });

  addEdge(graph, spandrelAutoscale, 'image', unsharpMask, 'image');

  const { output, title } = buildPath({
    denoiseSize,
    destination,
    graph,
    model,
    negativePrompt,
    positivePrompt,
    projectSettings,
    seed,
    settings,
    unsharpMask,
  });

  addUpscaleMetadata(graph, output, settings, projectSettings, randDevice);

  return {
    backendGraph: graph,
    graph: toGraphContract(graph, title),
    negativePromptNodeId: 'negative_prompt',
    outputNodeId: 'upscale_output',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};
