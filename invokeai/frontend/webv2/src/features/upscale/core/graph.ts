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
  getCompelPromptNodeType,
  getActiveCompatibleLoras,
  getLoadedLoras,
  toGraphContract,
  toModelIdentifier,
} from '@features/generation/graph';
import { isBundledMainForBase } from '@features/generation/settings';

import type { UpscaleBase, UpscaleIndexClipping } from './settings';
import type { CompiledUpscaleGraph, UpscaleWidgetValues } from './types';

import {
  activeUpscaleComponents,
  coerceUpscaleScheduler,
  getUpscaleValidationReasons,
  upscaleArchitectureFor,
  upscaleDenoiseDimension,
  wiresUpscaleNegativePrompt,
} from './settings';

/** The legacy formula: creativity -10..10 to a value-clipped start of 0.998..0. */
export const getUpscaleDenoisingStart = (creativity: number): number => ((creativity * -1 + 10) * 4.99) / 100;

/**
 * The start an index-clipping node needs to enter where a value-clipping one would.
 *
 * Creativity is defined by where FLUX enters: `flux_denoise` clips the schedule by *value*, so a start
 * of 0.499 enters at sigma 0.501 whatever the shift or step count. Z-Image and Anima clip by *index*
 * instead, so the same number enters far noisier there -- 0.499 starts Z-Image at sigma 0.78. All of
 * them build `sigma = shift * t / (1 + (shift - 1) * t)` at `t = 1 - i / steps`. This picks the `i`
 * whose sigma is nearest `1 - valueStart`. Z-Image and Anima begin at `i = int(start * steps)`, so the
 * start aims at the middle of that step, where float error in the node's product cannot floor it one
 * step short. Krea-2 rounds instead, half to even, so its start is the step itself. This goes away when
 * those nodes clip by value; `test_upscale_index_clipped_starts.py` fails when they do.
 */
export const getIndexClippedDenoisingStart = (
  valueStart: number,
  steps: number,
  shift: number,
  rounding: UpscaleIndexClipping['rounding']
): number => {
  const target = 1 - valueStart;
  let entryIndex = 0;
  let entryDistance = Number.POSITIVE_INFINITY;

  // The last index would leave nothing to denoise, so it is never a candidate.
  for (let index = 0; index < steps; index += 1) {
    const t = 1 - index / steps;
    const distance = Math.abs((shift * t) / (1 + (shift - 1) * t) - target);

    if (distance < entryDistance) {
      entryIndex = index;
      entryDistance = distance;
    }
  }

  if (entryIndex === 0) {
    return 0;
  }

  return rounding === 'floor' ? (entryIndex + 0.5) / steps : entryIndex / steps;
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

/**
 * Z-Image takes one control, not SD's pair. At Structure 0 it sits at the bottom of the adapter's
 * recommended 0.65-0.80 and lets go for the last quarter of the pass, where the detail is drawn;
 * the slider runs it from a light early nudge (0.3, first half) to holding the whole pass at 1.0.
 */
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
  const components = activeUpscaleComponents(settings);
  // FLUX calls this guidance, not CFG, and the encoders have to be recalled with the rest.
  const guidanceField =
    architecture?.negativePrompt === 'never' ? { guidance: settings.cfgScale } : { cfg_scale: settings.cfgScale };
  const metadata = addNode(graph, {
    ...guidanceField,
    ...(components.t5EncoderModel ? { t5_encoder: toModelIdentifier(components.t5EncoderModel) } : {}),
    ...(components.clipEmbedModel ? { clip_embed_model: toModelIdentifier(components.clipEmbedModel) } : {}),
    ...(components.qwen3EncoderModel ? { qwen3_encoder: toModelIdentifier(components.qwen3EncoderModel) } : {}),
    ...(components.qwen35EncoderModel ? { qwen3_5_encoder: toModelIdentifier(components.qwen35EncoderModel) } : {}),
    ...(components.qwen3VlEncoderModel ? { qwen3_vl_encoder: toModelIdentifier(components.qwen3VlEncoderModel) } : {}),
    creativity: settings.creativity,
    id: 'core_metadata',
    // Runs after the path builder, so the graph already holds the LoRAs this architecture loads.
    loras: getLoadedLoras(graph),
    model: toModelIdentifier(settings.model),
    rand_device: randDevice,
    scheduler: coerceUpscaleScheduler(settings.model, settings.scheduler),
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
  if (wiresUpscaleNegativePrompt(settings.model, settings.cfgScale)) {
    addEdge(graph, graph.nodes.negative_prompt, 'value', metadata, 'negative_prompt');
  }
  addEdge(graph, graph.nodes.seed, 'value', metadata, 'seed');
  addEdge(graph, frame, 'width', metadata, 'width');
  addEdge(graph, frame, 'height', metadata, 'height');
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
  /** The encoders this main model reads. A field it has no slot for is null here, whatever the widget holds. */
  components: ReturnType<typeof activeUpscaleComponents>;
  destination: ResultDestination;
  projectSettings: UpscaleProjectSettings;
  positivePrompt: BackendInvocationContract;
  negativePrompt: BackendInvocationContract;
  seed: BackendInvocationContract;
  /**
   * The enlarged image, at whatever size Spandrel produced. That is not always `denoiseSize`: a
   * model that does not enlarge makes Spandrel ignore the scale. See `fitFrameToDenoiseSize`.
   */
  frame: BackendInvocationContract;
  /** The frame size the second pass will run at, already floored to the architecture's grid. */
  denoiseSize: { width: number; height: number };
  /** Where the pass starts, already translated into this architecture's denoise node's terms. */
  denoisingStart: number;
}

interface UpscalePathResult {
  /** The node whose image is the result; the caller hangs metadata off it. */
  output: BackendInvocationContract;
  /**
   * The image the pass encoded, which the result matches in size: Spandrel's frame, or the one
   * resized to the denoise size. The metadata records its size, so Recall restores the real one.
   */
  frame: BackendInvocationContract;
  /** Names the pass in the queue, so "multi-diffusion" is not claimed where none runs. */
  title: string;
}

type UpscalePathBuilder = (context: UpscalePathContext) => UpscalePathResult;

/**
 * Resize the frame to the size a builder states as a literal on its denoise node.
 *
 * The literal is computed from the scale, but Spandrel ignores the scale for a model that does not
 * enlarge (a 1x restoration model), and it rounds to a multiple of 8 where `flux_denoise` and
 * `z_image_denoise` take multiples of 16. Either way the latents would disagree with the node's size,
 * and the pass would fail -- after the upscale has run. SD needs none of this: its sizes follow the
 * frame through edges.
 */
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

/**
 * FLUX.1 has no tile ControlNet, so the second pass is anchored only by where it starts on the
 * schedule. Both VAE ends tile: the frame handed to `i2l` is the largest in the graph, and an
 * untiled encode of it does not fit on consumer hardware.
 */
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

/**
 * Z-Image runs the pass whole like FLUX.1, both VAE ends tiled. Its tile ControlNet is optional: a
 * pass that does not tile has no seams to hold together, so the ControlNet only lets Structure pull
 * the result back toward the enlarged frame. The ControlNet encodes that frame inside the denoise,
 * at the same size as `i2l` does, so it is tiled the same way.
 */
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

/**
 * Anima has no tile ControlNet (only LLLite adapters), so like FLUX.1 the pass is anchored by where
 * it starts. Its decode has no tiling switch: `anima_l2i` tiles on its own once an untiled decode
 * would take most of the card.
 */
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

/**
 * Krea-2 runs the pass whole like FLUX.1, both VAE ends tiled. It encodes and decodes with the
 * Qwen-Image VAE through that family's nodes, as Generate and Canvas do. `krea2_denoise` picks its own
 * shift, fixed for Turbo and by size for Raw, so the graph sends none.
 */
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

/**
 * Z-Image's, Anima's and Krea-2's text encoders take their encoders on named fields and hand the denoise a
 * collected list; the negative branch exists only where the caller decided it is read.
 */
const addPromptConditioning = (
  graph: BackendGraphContract,
  {
    denoise,
    encoderFields,
    encoderType,
    negativePrompt,
    positivePrompt,
  }: {
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
    const collect = addNode(graph, { id: `${prefix}_cond_collect`, type: 'collect' });

    for (const { field, source } of encoderFields) {
      addEdge(graph, source, field, encoder, field);
    }
    addEdge(graph, prompt, 'value', encoder, 'prompt');
    addEdge(graph, encoder, 'conditioning', collect, 'item');
    addEdge(graph, collect, 'collection', denoise, conditioningField);
  }
};

/**
 * Keyed by the same bases as `UPSCALE_ARCHITECTURES`, and `satisfies` makes that a compile error to
 * break: declaring an architecture supported without giving it a builder will not type-check.
 */
const UPSCALE_PATH_BUILDERS = {
  anima: buildAnimaUpscalePath,
  flux: buildFluxUpscalePath,
  'krea-2': buildKrea2UpscalePath,
  'sd-1': buildSdUpscalePath,
  sdxl: buildSdUpscalePath,
  'z-image': buildZImageUpscalePath,
} satisfies Record<UpscaleBase, UpscalePathBuilder>;

export const compileUpscaleGraph = (
  settings: UpscaleWidgetValues,
  destination: ResultDestination,
  projectSettings: UpscaleProjectSettings,
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
        settings.steps,
        clipping.shift(model, denoiseSize),
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
