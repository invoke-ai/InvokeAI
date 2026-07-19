import type {
  BackendGraphContract,
  BackendInvocationContract,
  GenerationProjectSettings,
  ResultDestination,
} from '@features/generation/contracts';

import {
  addLoraCollectionLoader,
  addEdge,
  addNode,
  getActiveCompatibleLoras,
  toGraphContract,
  toModelIdentifier,
} from '@features/generation/graph';
import { coerceSchedulerForGraph } from '@features/generation/settings';

import type { CompiledUpscaleGraph, UpscaleWidgetValues } from './types';

import { getUpscaleOutputDimensions, getUpscaleValidationReasons } from './settings';

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
  projectSettings: GenerationProjectSettings
): void => {
  if (!settings.model || !settings.upscaleModel || !settings.inputImage) {
    return;
  }

  const dimensions = getUpscaleOutputDimensions(settings.inputImage, settings.scale);
  const activeLoras = getActiveCompatibleLoras(
    {
      ...settings,
      aspectRatioId: 'Free',
      aspectRatioIsLocked: false,
      aspectRatioValue: dimensions.width / dimensions.height,
      cfgRescaleMultiplier: 0,
      colorCompensation: false,
      componentSourceModel: null,
      height: dimensions.height,
      modelKey: settings.model.key,
      negativePromptHeightPx: 56,
      positivePromptHeightPx: 96,
      referenceImages: [],
      seamlessXAxis: false,
      seamlessYAxis: false,
      t5EncoderModel: null,
      clipEmbedModel: null,
      clipLEmbedModel: null,
      clipGEmbedModel: null,
      qwen3EncoderModel: null,
      qwenVLEncoderModel: null,
      width: dimensions.width,
    },
    settings.model
  );
  const metadata = addNode(graph, {
    cfg_scale: settings.cfgScale,
    creativity: settings.creativity,
    id: 'core_metadata',
    loras: activeLoras.map((lora) => ({ model: toModelIdentifier(lora.model), weight: lora.weight })),
    model: toModelIdentifier(settings.model),
    rand_device: projectSettings.useCpuNoise ? 'cpu' : 'cuda',
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
  addEdge(graph, graph.nodes.negative_prompt, 'value', metadata, 'negative_prompt');
  addEdge(graph, graph.nodes.seed, 'value', metadata, 'seed');
  addEdge(graph, graph.nodes.spandrel_autoscale, 'width', metadata, 'width');
  addEdge(graph, graph.nodes.spandrel_autoscale, 'height', metadata, 'height');
  addEdge(graph, metadata, 'metadata', output, 'metadata');
};

export const compileUpscaleGraph = (
  settings: UpscaleWidgetValues,
  destination: ResultDestination,
  projectSettings: GenerationProjectSettings
): CompiledUpscaleGraph => {
  const reasons = getUpscaleValidationReasons(settings);

  if (reasons.length > 0) {
    throw new Error(reasons[0]);
  }

  const { inputImage, model, tileControlnetModel, upscaleModel } = settings;

  if (!inputImage || !model || !tileControlnetModel || !upscaleModel) {
    throw new Error('Upscale settings are incomplete.');
  }

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

  addEdge(graph, spandrelAutoscale, 'image', unsharpMask, 'image');
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
  addUpscaleMetadata(graph, output, settings, projectSettings);

  return {
    backendGraph: graph,
    graph: toGraphContract(graph, `${model.name} multi-diffusion upscale`),
    negativePromptNodeId: 'negative_prompt',
    outputNodeId: 'upscale_output',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};
