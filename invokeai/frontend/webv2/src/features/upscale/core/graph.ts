import type {
  BackendGraphContract,
  BackendInvocationContract,
  GenerationProjectSettings,
  ImageWithDims,
  MainModelConfig,
  ResultDestination,
} from '@features/generation/contracts';

import {
  addLoraCollectionLoader,
  addEdge,
  addNode,
  addTransformerLoraCollectionLoader,
  getCompelPromptNodeType,
  getActiveCompatibleLoras,
  getLoadedLoras,
  getRecordedComponents,
  toGraphContract,
  toModelIdentifier,
} from '@features/generation/graph';
import { isBundledMainForBase } from '@features/generation/settings';

import type { UpscaleBase, UpscaleIndexClipping } from './settings';
import type { CompiledUpscaleGraph, SpandrelModelConfig, UpscaleWidgetValues } from './types';

import {
  activeUpscaleComponents,
  coerceUpscaleScheduler,
  getSpandrelNativeScale,
  getUpscaleValidationReasons,
  upscaleArchitectureFor,
  upscaleDenoiseDimension,
  wiresUpscaleNegativePrompt,
} from './settings';

/** The legacy formula: creativity -10..10 to a value-clipped start of 0.998..0. */
export const getUpscaleDenoisingStart = (creativity: number): number => ((creativity * -1 + 10) * 4.99) / 100;

/**
 * Creativity is defined by FLUX's value clipping; index-clipping nodes would enter far noisier at the same start.
 * Floor nodes aim mid-step so float error cannot drop a step; rounding nodes take the step itself.
 * Obsolete once those nodes clip by value; `test_upscale_index_clipped_starts.py` fails when they do.
 */
export const getIndexClippedDenoisingStart = (
  valueStart: number,
  sigmas: readonly number[],
  rounding: UpscaleIndexClipping['rounding']
): number => {
  const target = 1 - valueStart;
  let entryIndex = 0;
  let entryDistance = Number.POSITIVE_INFINITY;

  for (const [index, sigma] of sigmas.entries()) {
    const distance = Math.abs(sigma - target);

    if (distance < entryDistance) {
      entryIndex = index;
      entryDistance = distance;
    }
  }

  if (entryIndex === 0) {
    return 0;
  }

  return rounding === 'floor' ? (entryIndex + 0.5) / sigmas.length : entryIndex / sigmas.length;
};

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

/** Structure 0 sits at the adapter's recommended 0.65 and releases the last quarter; 1 holds the pass at 1.0. */
export const getZImageTileControlValues = (structure: number) => ({
  beginStepPercent: 0,
  controlContextScale: 0.65 + structure * 0.035,
  endStepPercent: 0.75 + structure * 0.025,
});

/** Upscale runs no output filters, so only the noise device reaches its graph. */
type UpscaleProjectSettings = Pick<GenerationProjectSettings, 'useCpuNoise'>;

const addUpscaleMetadata = (
  graph: BackendGraphContract,
  { frame, output }: UpscalePathResult,
  settings: UpscaleWidgetValues,
  projectSettings: UpscaleProjectSettings,
  randDevice: string
): void => {
  if (!settings.model || !settings.upscaleModel || !settings.inputImage) {
    return;
  }

  const architecture = upscaleArchitectureFor(settings.model);
  const tileControlModel = (graph.nodes.controlnet_1 ?? graph.nodes.tile_control)?.control_model;
  // FLUX calls this guidance, not CFG.
  const guidanceField =
    architecture?.negativePrompt === 'never' ? { guidance: settings.cfgScale } : { cfg_scale: settings.cfgScale };
  const metadata = addNode(graph, {
    ...guidanceField,
    // The loaders hold the encoders and the VAE, under the keys Generate records them with.
    ...getRecordedComponents(graph),
    // Only the SD path runs its encode and decode at a chosen precision.
    ...(graph.nodes.i2l?.type === 'i2l' ? { vae_precision: settings.vaePrecision } : {}),
    creativity: settings.creativity,
    id: 'core_metadata',
    // Runs after the path builder, so the graph already holds the LoRAs this architecture loads.
    loras: getLoadedLoras(graph),
    model: toModelIdentifier(settings.model),
    rand_device: randDevice,
    scheduler: coerceUpscaleScheduler(settings.model, settings.scheduler),
    steps: settings.steps,
    structure: settings.structure,
    // The ControlNet node the path added: the architecture's own tile control, when it takes one.
    ...(tileControlModel ? { tile_controlnet: tileControlModel } : {}),
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
  });

  addEdge(graph, graph.nodes.positive_prompt, 'value', metadata, 'positive_prompt');
  if (wiresUpscaleNegativePrompt(settings.model, settings.cfgScale)) {
    addEdge(graph, graph.nodes.negative_prompt, 'value', metadata, 'negative_prompt');
  }
  addEdge(graph, graph.nodes.seed, 'value', metadata, 'seed');
  addEdge(graph, frame, 'width', metadata, 'width');
  addEdge(graph, frame, 'height', metadata, 'height');
  addEdge(graph, metadata, 'metadata', output, 'metadata');
};

/** The shared prefix (prompts, seed, Spandrel, unsharp) is built once; builders add loader, conditioning and VAE. */
interface UpscalePathContext {
  graph: BackendGraphContract;
  settings: UpscaleWidgetValues;
  model: MainModelConfig;
  /** A field this main model has no slot for is null here, whatever the widget holds. */
  components: ReturnType<typeof activeUpscaleComponents>;
  destination: ResultDestination;
  projectSettings: UpscaleProjectSettings;
  positivePrompt: BackendInvocationContract;
  negativePrompt: BackendInvocationContract;
  seed: BackendInvocationContract;
  /** Not always `denoiseSize`: Spandrel ignores the scale for a model that does not enlarge. */
  frame: BackendInvocationContract;
  denoiseSize: { width: number; height: number };
  denoisingStart: number;
}

interface UpscalePathResult {
  output: BackendInvocationContract;
  /** Spandrel's frame or its resize; metadata records its size, so Recall restores the real one. */
  frame: BackendInvocationContract;
  /** Names the pass in the queue, so "multi-diffusion" is not claimed where none runs. */
  title: string;
}

type UpscalePathBuilder = (context: UpscalePathContext) => UpscalePathResult;

/** Spandrel ignores scale for 1x models and rounds to 8 where nodes need 16; a mismatch fails after upscaling. */
const fitFrameToDenoiseSize = (
  graph: BackendGraphContract,
  frame: BackendInvocationContract,
  denoiseSize: { width: number; height: number }
): BackendInvocationContract => {
  const fitted = addNode(graph, {
    height: denoiseSize.height,
    id: 'fit_to_grid',
    resample_mode: 'bicubic',
    type: 'img_resize',
    width: denoiseSize.width,
  });

  addEdge(graph, frame, 'image', fitted, 'image');

  return fitted;
};

const buildSdUpscalePath: UpscalePathBuilder = ({
  graph,
  settings,
  model,
  destination,
  projectSettings,
  positivePrompt,
  negativePrompt,
  seed,
  frame,
  denoisingStart,
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
    denoising_start: denoisingStart,
    id: 'tiled_multidiffusion_denoise_latents',
    scheduler: coerceUpscaleScheduler(model, settings.scheduler),
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
  const compelType = getCompelPromptNodeType(model.base);
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
  addEdge(graph, frame, 'width', noise, 'width');
  addEdge(graph, frame, 'height', noise, 'height');
  addEdge(graph, frame, 'image', imageToLatents, 'image');

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

  addEdge(graph, frame, 'image', controlNet1, 'image');
  addEdge(graph, frame, 'image', controlNet2, 'image');
  addEdge(graph, controlNet1, 'control', controlNetCollector, 'item');
  addEdge(graph, controlNet2, 'control', controlNetCollector, 'item');
  addEdge(graph, controlNetCollector, 'collection', denoise, 'control');

  return { frame, output, title: `${model.name} multi-diffusion upscale` };
};

/** No tile ControlNet: anchored only by its start. Both VAE ends tile; an untiled encode exceeds consumer GPUs. */
const buildFluxUpscalePath: UpscalePathBuilder = ({
  graph,
  settings,
  model,
  components,
  destination,
  positivePrompt,
  seed,
  frame,
  denoiseSize,
  denoisingStart,
}) => {
  const modelLoader = addNode(graph, {
    clip_embed_model: components.clipEmbedModel ? toModelIdentifier(components.clipEmbedModel) : undefined,
    id: 'model_loader',
    model: toModelIdentifier(model),
    t5_encoder_model: components.t5EncoderModel ? toModelIdentifier(components.t5EncoderModel) : undefined,
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
  const imageToLatents = addNode(graph, {
    id: 'i2l',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'flux_vae_encode',
  });
  const denoise = addNode(graph, {
    denoising_end: 1,
    denoising_start: denoisingStart,
    guidance: settings.cfgScale,
    height: denoiseSize.height,
    id: 'denoise_latents',
    num_steps: settings.steps,
    scheduler: coerceUpscaleScheduler(model, settings.scheduler),
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

  const fitted = fitFrameToDenoiseSize(graph, frame, denoiseSize);

  addEdge(graph, fitted, 'image', imageToLatents, 'image');
  addEdge(graph, modelLoader, 'vae', imageToLatents, 'vae');
  addEdge(graph, imageToLatents, 'latents', denoise, 'latents');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addEdge(graph, modelLoader, 'vae', output, 'vae');

  return { frame: fitted, output, title: `${model.name} upscale` };
};

/** As in Generate, CFG stays off and the slider drives [dev]'s distilled guidance; Klein ignores it. */
const buildFlux2UpscalePath: UpscalePathBuilder = ({
  graph,
  settings,
  model,
  components,
  destination,
  positivePrompt,
  seed,
  frame,
  denoiseSize,
  denoisingStart,
}) => {
  const isDev = model.variant === 'dev';
  const encoderField = isDev ? 'mistral_encoder' : 'qwen3_encoder';
  const vaeModel = settings.vae ? toModelIdentifier(settings.vae) : undefined;
  const modelLoader = addNode(
    graph,
    isDev
      ? {
          id: 'model_loader',
          mistral_encoder_model: components.mistralEncoderModel
            ? toModelIdentifier(components.mistralEncoderModel)
            : undefined,
          model: toModelIdentifier(model),
          type: 'flux2_dev_model_loader',
          vae_model: vaeModel,
        }
      : {
          id: 'model_loader',
          model: toModelIdentifier(model),
          qwen3_encoder_model: components.qwen3EncoderModel
            ? toModelIdentifier(components.qwen3EncoderModel)
            : undefined,
          type: 'flux2_klein_model_loader',
          vae_model: vaeModel,
        }
  );
  const activeLoras = getActiveCompatibleLoras({ loras: settings.loras }, model);
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(
        graph,
        activeLoras,
        isDev ? 'flux2_dev_lora_collection_loader' : 'flux2_klein_lora_collection_loader',
        modelLoader,
        ['transformer', encoderField]
      )
    : modelLoader;
  const posCond = addNode(graph, {
    id: 'pos_cond',
    type: isDev ? 'flux2_dev_text_encoder' : 'flux2_klein_text_encoder',
  });
  const posCondCollect = addNode(graph, { id: 'pos_cond_collect', type: 'collect' });
  const imageToLatents = addNode(graph, {
    id: 'i2l',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'flux2_vae_encode',
  });
  const denoise = addNode(graph, {
    denoising_end: 1,
    denoising_start: denoisingStart,
    guidance: settings.cfgScale,
    height: denoiseSize.height,
    id: 'denoise_latents',
    num_steps: settings.steps,
    scheduler: coerceUpscaleScheduler(model, settings.scheduler),
    type: 'flux2_denoise',
    width: denoiseSize.width,
  });
  const output = addNode(graph, {
    id: 'upscale_output',
    is_intermediate: destination === 'canvas',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'flux2_vae_decode',
    use_cache: false,
  });

  addEdge(graph, loraSource, encoderField, posCond, encoderField);
  addEdge(graph, modelLoader, 'max_seq_len', posCond, 'max_seq_len');
  addEdge(graph, positivePrompt, 'value', posCond, 'prompt');
  addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
  addEdge(graph, posCondCollect, 'collection', denoise, 'positive_text_conditioning');
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, seed, 'value', denoise, 'seed');

  const fitted = fitFrameToDenoiseSize(graph, frame, denoiseSize);

  addEdge(graph, fitted, 'image', imageToLatents, 'image');
  addEdge(graph, modelLoader, 'vae', imageToLatents, 'vae');
  addEdge(graph, imageToLatents, 'latents', denoise, 'latents');
  // The denoise normalizes the encoded latents with this VAE's BN statistics.
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addEdge(graph, modelLoader, 'vae', output, 'vae');

  return { frame: fitted, output, title: `${model.name} upscale` };
};

/** The tile ControlNet is optional (no tiles, no seams); it encodes the frame like `i2l`, so tiles the same way. */
const buildZImageUpscalePath: UpscalePathBuilder = ({
  graph,
  settings,
  model,
  components,
  destination,
  positivePrompt,
  negativePrompt,
  seed,
  frame,
  denoiseSize,
  denoisingStart,
}) => {
  const fitted = fitFrameToDenoiseSize(graph, frame, denoiseSize);
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model: toModelIdentifier(model),
    qwen3_encoder_model: components.qwen3EncoderModel ? toModelIdentifier(components.qwen3EncoderModel) : undefined,
    // A Diffusers or complete SDNQ install supplies its own VAE and Qwen3 where none is chosen.
    qwen3_source_model: isBundledMainForBase('z-image')(model) ? toModelIdentifier(model) : undefined,
    type: 'z_image_model_loader',
    vae_model: settings.vae ? toModelIdentifier(settings.vae) : undefined,
  });
  const activeLoras = getActiveCompatibleLoras({ loras: settings.loras }, model);
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'z_image_lora_collection_loader', modelLoader, [
        'transformer',
        'qwen3_encoder',
      ])
    : modelLoader;
  const imageToLatents = addNode(graph, {
    id: 'i2l',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'z_image_i2l',
  });
  const denoise = addNode(graph, {
    denoising_end: 1,
    denoising_start: denoisingStart,
    guidance_scale: settings.cfgScale,
    height: denoiseSize.height,
    id: 'denoise_latents',
    scheduler: coerceUpscaleScheduler(model, settings.scheduler),
    steps: settings.steps,
    type: 'z_image_denoise',
    width: denoiseSize.width,
  });
  const output = addNode(graph, {
    id: 'upscale_output',
    is_intermediate: destination === 'canvas',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'z_image_l2i',
    use_cache: false,
  });

  addPromptConditioning(graph, {
    denoise,
    encoderFields: [{ field: 'qwen3_encoder', source: loraSource }],
    encoderType: 'z_image_text_encoder',
    negativePrompt: wiresUpscaleNegativePrompt(model, settings.cfgScale) ? negativePrompt : null,
    positivePrompt,
  });
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, seed, 'value', denoise, 'seed');
  addEdge(graph, fitted, 'image', imageToLatents, 'image');
  addEdge(graph, modelLoader, 'vae', imageToLatents, 'vae');
  addEdge(graph, imageToLatents, 'latents', denoise, 'latents');
  // The denoise encodes its control image itself, with this VAE.
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addEdge(graph, modelLoader, 'vae', output, 'vae');

  if (settings.tileControlnetModel) {
    const control = getZImageTileControlValues(settings.structure);
    const tileControl = addNode(graph, {
      begin_step_percent: control.beginStepPercent,
      control_context_scale: control.controlContextScale,
      control_model: toModelIdentifier(settings.tileControlnetModel),
      end_step_percent: control.endStepPercent,
      id: 'tile_control',
      tile_size: settings.tileSize,
      tiled: true,
      type: 'z_image_control',
    });

    addEdge(graph, fitted, 'image', tileControl, 'image');
    addEdge(graph, tileControl, 'control', denoise, 'control');
  }

  return { frame: fitted, output, title: `${model.name} upscale` };
};

/** No tile ControlNet (only LLLite). `anima_l2i` tiles on its own once an untiled decode would fill the card. */
const buildAnimaUpscalePath: UpscalePathBuilder = ({
  graph,
  settings,
  model,
  components,
  destination,
  positivePrompt,
  negativePrompt,
  seed,
  frame,
  denoiseSize,
  denoisingStart,
}) => {
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model: toModelIdentifier(model),
    qwen3_5_encoder_model: components.qwen35EncoderModel ? toModelIdentifier(components.qwen35EncoderModel) : undefined,
    qwen3_encoder_model: components.qwen3EncoderModel ? toModelIdentifier(components.qwen3EncoderModel) : undefined,
    type: 'anima_model_loader',
    vae_model: settings.vae ? toModelIdentifier(settings.vae) : undefined,
  });
  const activeLoras = getActiveCompatibleLoras({ loras: settings.loras }, model);
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'anima_lora_collection_loader', modelLoader, [
        'transformer',
        'qwen3_encoder',
      ])
    : modelLoader;
  const imageToLatents = addNode(graph, {
    id: 'i2l',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'anima_i2l',
  });
  const denoise = addNode(graph, {
    denoising_end: 1,
    denoising_start: denoisingStart,
    guidance_scale: settings.cfgScale,
    height: denoiseSize.height,
    id: 'denoise_latents',
    scheduler: coerceUpscaleScheduler(model, settings.scheduler),
    steps: settings.steps,
    type: 'anima_denoise',
    width: denoiseSize.width,
  });
  const output = addNode(graph, {
    id: 'upscale_output',
    is_intermediate: destination === 'canvas',
    type: 'anima_l2i',
    use_cache: false,
  });

  addPromptConditioning(graph, {
    denoise,
    encoderFields: [
      { field: 'qwen3_encoder', source: loraSource },
      // Straight from the loader: the LoRA collection loader passes Qwen3.5 through untouched.
      ...(components.qwen35EncoderModel ? [{ field: 'qwen3_5_encoder', source: modelLoader }] : []),
    ],
    encoderType: 'anima_text_encoder',
    negativePrompt: wiresUpscaleNegativePrompt(model, settings.cfgScale) ? negativePrompt : null,
    positivePrompt,
  });
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, seed, 'value', denoise, 'seed');
  const fitted = fitFrameToDenoiseSize(graph, frame, denoiseSize);

  addEdge(graph, fitted, 'image', imageToLatents, 'image');
  addEdge(graph, modelLoader, 'vae', imageToLatents, 'vae');
  addEdge(graph, imageToLatents, 'latents', denoise, 'latents');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addEdge(graph, modelLoader, 'vae', output, 'vae');

  return { frame: fitted, output, title: `${model.name} upscale` };
};

/** Uses the Qwen-Image VAE nodes, as Generate does. `krea2_denoise` picks its own shift, so none is sent. */
const buildKrea2UpscalePath: UpscalePathBuilder = ({
  graph,
  settings,
  model,
  components,
  destination,
  positivePrompt,
  negativePrompt,
  seed,
  frame,
  denoiseSize,
  denoisingStart,
}) => {
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model: toModelIdentifier(model),
    qwen3_vl_encoder_model: components.qwen3VlEncoderModel
      ? toModelIdentifier(components.qwen3VlEncoderModel)
      : undefined,
    type: 'krea2_model_loader',
    vae_model: settings.vae ? toModelIdentifier(settings.vae) : undefined,
  });
  const activeLoras = getActiveCompatibleLoras({ loras: settings.loras }, model);
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'krea2_lora_collection_loader', modelLoader, [
        'transformer',
        'qwen3_vl_encoder',
      ])
    : modelLoader;
  const imageToLatents = addNode(graph, {
    id: 'i2l',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'qwen_image_i2l',
  });
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    denoising_end: 1,
    denoising_start: denoisingStart,
    height: denoiseSize.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'krea2_denoise',
    width: denoiseSize.width,
  });
  const output = addNode(graph, {
    id: 'upscale_output',
    is_intermediate: destination === 'canvas',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'qwen_image_l2i',
    use_cache: false,
  });

  addPromptConditioning(graph, {
    denoise,
    encoderFields: [{ field: 'qwen3_vl_encoder', source: loraSource }],
    encoderType: 'krea2_text_encoder',
    negativePrompt: wiresUpscaleNegativePrompt(model, settings.cfgScale) ? negativePrompt : null,
    positivePrompt,
  });
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, seed, 'value', denoise, 'seed');

  const fitted = fitFrameToDenoiseSize(graph, frame, denoiseSize);

  addEdge(graph, fitted, 'image', imageToLatents, 'image');
  addEdge(graph, modelLoader, 'vae', imageToLatents, 'vae');
  addEdge(graph, imageToLatents, 'latents', denoise, 'latents');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addEdge(graph, modelLoader, 'vae', output, 'vae');

  return { frame: fitted, output, title: `${model.name} upscale` };
};

/** No shift is sent, so `qwen_image_denoise` keeps its dynamic one, which Creativity is translated against. */
const buildQwenImageUpscalePath: UpscalePathBuilder = ({
  graph,
  settings,
  model,
  components,
  destination,
  positivePrompt,
  negativePrompt,
  seed,
  frame,
  denoiseSize,
  denoisingStart,
}) => {
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model: toModelIdentifier(model),
    qwen_vl_encoder_model: components.qwenVlEncoderModel ? toModelIdentifier(components.qwenVlEncoderModel) : undefined,
    type: 'qwen_image_model_loader',
    vae_model: settings.vae ? toModelIdentifier(settings.vae) : undefined,
  });
  const activeLoras = getActiveCompatibleLoras({ loras: settings.loras }, model);
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'qwen_image_lora_collection_loader', modelLoader, [
        'transformer',
        'qwen_vl_encoder',
      ])
    : modelLoader;
  const imageToLatents = addNode(graph, {
    id: 'i2l',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'qwen_image_i2l',
  });
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    denoising_end: 1,
    denoising_start: denoisingStart,
    height: denoiseSize.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'qwen_image_denoise',
    width: denoiseSize.width,
  });
  const output = addNode(graph, {
    id: 'upscale_output',
    is_intermediate: destination === 'canvas',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'qwen_image_l2i',
    use_cache: false,
  });

  addPromptConditioning(graph, {
    collect: false,
    denoise,
    encoderFields: [{ field: 'qwen_vl_encoder', source: loraSource }],
    encoderType: 'qwen_image_text_encoder',
    negativePrompt: wiresUpscaleNegativePrompt(model, settings.cfgScale) ? negativePrompt : null,
    positivePrompt,
  });
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, seed, 'value', denoise, 'seed');

  const fitted = fitFrameToDenoiseSize(graph, frame, denoiseSize);

  addEdge(graph, fitted, 'image', imageToLatents, 'image');
  addEdge(graph, modelLoader, 'vae', imageToLatents, 'vae');
  addEdge(graph, imageToLatents, 'latents', denoise, 'latents');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addEdge(graph, modelLoader, 'vae', output, 'vae');

  return { frame: fitted, output, title: `${model.name} upscale` };
};

/** Qwen Image's and MOSS-Image's denoise take one conditioning; the others a collected list. */
const addPromptConditioning = (
  graph: BackendGraphContract,
  {
    collect = true,
    denoise,
    encoderFields,
    encoderType,
    negativePrompt,
    positivePrompt,
  }: {
    collect?: boolean;
    denoise: BackendInvocationContract;
    encoderFields: readonly { field: string; source: BackendInvocationContract }[];
    encoderType: string;
    negativePrompt: BackendInvocationContract | null;
    positivePrompt: BackendInvocationContract;
  }
): void => {
  const branches = [
    { conditioningField: 'positive_conditioning', prefix: 'pos', prompt: positivePrompt },
    ...(negativePrompt ? [{ conditioningField: 'negative_conditioning', prefix: 'neg', prompt: negativePrompt }] : []),
  ];

  for (const { conditioningField, prefix, prompt } of branches) {
    const encoder = addNode(graph, { id: `${prefix}_cond`, type: encoderType });

    for (const { field, source } of encoderFields) {
      addEdge(graph, source, field, encoder, field);
    }
    addEdge(graph, prompt, 'value', encoder, 'prompt');

    if (collect) {
      const collector = addNode(graph, { id: `${prefix}_cond_collect`, type: 'collect' });

      addEdge(graph, encoder, 'conditioning', collector, 'item');
      addEdge(graph, collector, 'collection', denoise, conditioningField);
    } else {
      addEdge(graph, encoder, 'conditioning', denoise, conditioningField);
    }
  }
};

/**
 * `moss_image_denoise` uses the FLUX.2 VAE's raw latent layout, so FLUX.2's tiled encode/decode nodes fit. Its LoRAs
 * load on the transformer only.
 * Generate's layout prompt is left out: it only compiles layout drafts or canvas regions.
 */
const buildMossImageUpscalePath: UpscalePathBuilder = ({
  graph,
  settings,
  model,
  components,
  destination,
  positivePrompt,
  negativePrompt,
  seed,
  frame,
  denoiseSize,
  denoisingStart,
}) => {
  const modelLoader = addNode(graph, {
    id: 'model_loader',
    model: toModelIdentifier(model),
    qwen3_5_encoder_model: components.qwen35EncoderModel ? toModelIdentifier(components.qwen35EncoderModel) : undefined,
    type: 'moss_image_model_loader',
    vae_model: settings.vae ? toModelIdentifier(settings.vae) : undefined,
  });
  // MOSS-Image LoRAs train the transformer only; the VAE and the Qwen3.5 encoder stay on the model loader.
  const activeLoras = getActiveCompatibleLoras({ loras: settings.loras }, model);
  const loraSource = activeLoras.length
    ? addTransformerLoraCollectionLoader(graph, activeLoras, 'moss_image_lora_collection_loader', modelLoader, [
        'transformer',
      ])
    : modelLoader;
  const imageToLatents = addNode(graph, {
    id: 'i2l',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'flux2_vae_encode',
  });
  const denoise = addNode(graph, {
    cfg_scale: settings.cfgScale,
    denoising_end: 1,
    denoising_start: denoisingStart,
    height: denoiseSize.height,
    id: 'denoise_latents',
    steps: settings.steps,
    type: 'moss_image_denoise',
    width: denoiseSize.width,
  });
  const output = addNode(graph, {
    id: 'upscale_output',
    is_intermediate: destination === 'canvas',
    tile_size: settings.tileSize,
    tiled: true,
    type: 'flux2_vae_decode',
    use_cache: false,
  });

  addPromptConditioning(graph, {
    collect: false,
    denoise,
    encoderFields: [{ field: 'qwen3_5_encoder', source: modelLoader }],
    encoderType: 'moss_image_text_encoder',
    negativePrompt: wiresUpscaleNegativePrompt(model, settings.cfgScale) ? negativePrompt : null,
    positivePrompt,
  });
  addEdge(graph, loraSource, 'transformer', denoise, 'transformer');
  addEdge(graph, seed, 'value', denoise, 'seed');
  const fitted = fitFrameToDenoiseSize(graph, frame, denoiseSize);

  addEdge(graph, fitted, 'image', imageToLatents, 'image');
  addEdge(graph, modelLoader, 'vae', imageToLatents, 'vae');
  addEdge(graph, imageToLatents, 'latents', denoise, 'latents');
  // The denoise reads this VAE's BatchNorm statistics, which define the latent space it works in.
  addEdge(graph, modelLoader, 'vae', denoise, 'vae');
  addEdge(graph, denoise, 'latents', output, 'latents');
  addEdge(graph, modelLoader, 'vae', output, 'vae');

  return { frame: fitted, output, title: `${model.name} upscale` };
};

/** `satisfies` makes a supported architecture without a builder a type error. */
const UPSCALE_PATH_BUILDERS = {
  anima: buildAnimaUpscalePath,
  flux: buildFluxUpscalePath,
  flux2: buildFlux2UpscalePath,
  'krea-2': buildKrea2UpscalePath,
  'moss-image': buildMossImageUpscalePath,
  'qwen-image': buildQwenImageUpscalePath,
  'sd-1': buildSdUpscalePath,
  sdxl: buildSdUpscalePath,
  'z-image': buildZImageUpscalePath,
} satisfies Record<UpscaleBase, UpscalePathBuilder>;

export interface ModelOnlyUpscaleRequest {
  image: ImageWithDims;
  upscaleModel: SpandrelModelConfig;
  /** `'native'` runs the model once at its own scale. */
  scale: number | 'native';
  destination: ResultDestination;
  sourceMetadata: Record<string, unknown> | null;
}

/**
 * The upscaler alone. Source metadata sits on the output node as a literal: a `core_metadata` node would
 * validate old keys against today's fields and overwrite the source's `metadata_version`.
 */
export const compileModelOnlyUpscaleGraph = ({
  destination,
  image,
  scale,
  sourceMetadata,
  upscaleModel,
}: ModelOnlyUpscaleRequest): CompiledUpscaleGraph => {
  const graph: BackendGraphContract = { edges: [], id: 'upscale-graph', nodes: {} };
  // The queue's generate submission writes the batch's prompts and seed into these; nothing in this graph reads them.
  addNode(graph, { id: 'positive_prompt', type: 'string' });
  addNode(graph, { id: 'negative_prompt', type: 'string' });
  addNode(graph, { id: 'seed', type: 'integer' });

  const upscaleScale = scale === 'native' ? getSpandrelNativeScale(upscaleModel) : scale;
  // A source that was itself upscaled names its own upscale; none of those keys may survive into this one.
  const {
    upscale_initial_image: _initialImage,
    upscale_model: _model,
    upscale_scale: _scale,
    ...copiedMetadata
  } = sourceMetadata ?? {};
  const metadata = {
    ...copiedMetadata,
    upscale_initial_image: { height: image.height, image_name: image.image_name, width: image.width },
    upscale_model: toModelIdentifier(upscaleModel),
    ...(upscaleScale === null ? {} : { upscale_scale: upscaleScale }),
  };

  addNode(graph, {
    id: 'upscale_output',
    image: { image_name: image.image_name },
    image_to_image_model: toModelIdentifier(upscaleModel),
    is_intermediate: destination === 'canvas',
    metadata,
    // Matches the Refine graph's first pass; only an upscaler of unknown native scale runs a plain pass.
    ...(upscaleScale === null
      ? { type: 'spandrel_image_to_image' }
      : { fit_to_multiple_of_8: true, scale: upscaleScale, type: 'spandrel_image_to_image_autoscale' }),
    // A cached output would hand back the earlier result's image instead of a new one.
    use_cache: false,
  });

  return {
    backendGraph: graph,
    graph: toGraphContract(graph, `${upscaleModel.name} upscale`),
    negativePromptNodeId: 'negative_prompt',
    outputNodeId: 'upscale_output',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};

/** `sourceMetadata` is read only in Model only mode. */
export const compileUpscaleGraph = (
  settings: UpscaleWidgetValues,
  destination: ResultDestination,
  projectSettings: UpscaleProjectSettings,
  randDevice = projectSettings.useCpuNoise ? 'cpu' : 'cuda',
  sourceMetadata: Record<string, unknown> | null = null
): CompiledUpscaleGraph => {
  const reasons = getUpscaleValidationReasons(settings);

  if (reasons.length > 0) {
    throw new Error(reasons[0]);
  }

  if (settings.mode === 'model-only') {
    if (!settings.inputImage || !settings.upscaleModel) {
      throw new Error('Upscale settings are incomplete.');
    }

    return compileModelOnlyUpscaleGraph({
      destination,
      image: settings.inputImage,
      scale: settings.scale,
      sourceMetadata,
      upscaleModel: settings.upscaleModel,
    });
  }

  const { inputImage, model, upscaleModel } = settings;

  if (!inputImage || !model || !upscaleModel) {
    throw new Error('Upscale settings are incomplete.');
  }

  const buildPath = UPSCALE_PATH_BUILDERS[model.base as UpscaleBase];

  if (!buildPath) {
    throw new Error(`Upscale does not support ${model.base} main models.`);
  }

  const architecture = upscaleArchitectureFor(model);
  const grid = architecture?.denoiseGrid ?? 8;
  const denoiseSize = {
    height: upscaleDenoiseDimension(inputImage.height, settings.scale, grid),
    width: upscaleDenoiseDimension(inputImage.width, settings.scale, grid),
  };
  const valueClippedStart = getUpscaleDenoisingStart(settings.creativity);
  const clipping = architecture?.indexClipping;
  const denoisingStart = clipping
    ? getIndexClippedDenoisingStart(
        valueClippedStart,
        clipping.sigmas(model, denoiseSize, settings.steps),
        clipping.rounding
      )
    : valueClippedStart;
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

  const path = buildPath({
    components: activeUpscaleComponents(settings),
    denoiseSize,
    denoisingStart,
    destination,
    frame: unsharpMask,
    graph,
    model,
    negativePrompt,
    positivePrompt,
    projectSettings,
    seed,
    settings,
  });

  addUpscaleMetadata(graph, path, settings, projectSettings, randDevice);

  return {
    backendGraph: graph,
    graph: toGraphContract(graph, path.title),
    negativePromptNodeId: 'negative_prompt',
    outputNodeId: 'upscale_output',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};
