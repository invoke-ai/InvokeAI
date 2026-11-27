import type {
  BackendGraphContract,
  BackendInvocationContract,
  GenerationProjectSettings,
  ImageWithDims,
  MainModelConfig,
  ResultDestination,
} from '@features/generation/contracts';
import type { FamilyPipelineSettings } from '@features/generation/graph';
import type { StepClipping } from '@features/generation/settings';

import {
  addEdge,
  addNode,
  buildImageToImagePipeline,
  getLoadedLoras,
  toGraphContract,
  toModelIdentifier,
} from '@features/generation/graph';
import { getDenoisingStartClipping } from '@features/generation/settings';

import type { UpscaleTileControlNet } from './settings';
import type { CompiledUpscaleGraph, SpandrelModelConfig, UpscaleWidgetValues } from './types';

import {
  activeUpscaleComponents,
  coerceUpscaleScheduler,
  getSpandrelNativeScale,
  getUpscaleValidationReasons,
  upscaleArchitectureFor,
  upscaleDenoiseDimension,
  upscaleDenoiseGrid,
  upscaleNegativePromptUsage,
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
  rounding: StepClipping['rounding']
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
  {
    frame,
    output,
    vaeModel,
  }: {
    /** The image the pass encoded, which the result matches in size, so Recall restores the real one. */
    frame: BackendInvocationContract;
    output: BackendInvocationContract;
    /** The VAE the pipeline loaded; the widget's pick is dropped where it is incompatible or the family ignores it. */
    vaeModel: UpscaleWidgetValues['vae'];
  },
  settings: UpscaleWidgetValues,
  randDevice: string
): void => {
  if (!settings.model || !settings.upscaleModel || !settings.inputImage) {
    return;
  }

  const components = activeUpscaleComponents(settings);
  // FLUX calls this guidance, not CFG, and the encoders have to be recalled with the rest.
  const guidanceField =
    upscaleNegativePromptUsage(settings.model) === 'never'
      ? { guidance: settings.cfgScale }
      : { cfg_scale: settings.cfgScale };
  const metadata = addNode(graph, {
    ...guidanceField,
    ...(components.t5EncoderModel ? { t5_encoder: toModelIdentifier(components.t5EncoderModel) } : {}),
    ...(components.clipEmbedModel ? { clip_embed_model: toModelIdentifier(components.clipEmbedModel) } : {}),
    ...(components.qwen3EncoderModel ? { qwen3_encoder: toModelIdentifier(components.qwen3EncoderModel) } : {}),
    ...(components.qwen35EncoderModel ? { qwen3_5_encoder: toModelIdentifier(components.qwen35EncoderModel) } : {}),
    ...(components.qwen3VlEncoderModel ? { qwen3_vl_encoder: toModelIdentifier(components.qwen3VlEncoderModel) } : {}),
    ...(components.mistralEncoderModel ? { mistral_encoder: toModelIdentifier(components.mistralEncoderModel) } : {}),
    // Generate records Qwen Image's encoder under its own key, and Remix reads it back from there.
    ...(components.qwenVlEncoderModel
      ? { qwen_image_qwen_vl_encoder: toModelIdentifier(components.qwenVlEncoderModel) }
      : {}),
    creativity: settings.creativity,
    id: 'core_metadata',
    // Runs after the pipeline, so the graph already holds the LoRAs this architecture loads.
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
    vae: vaeModel ? toModelIdentifier(vaeModel) : undefined,
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

/** The widget's values in the pipeline's terms. Upscale has no component source, and leaves CFG rescale off. */
const toPipelineSettings = (settings: UpscaleWidgetValues, model: MainModelConfig): FamilyPipelineSettings => {
  const components = activeUpscaleComponents(settings);

  return {
    cfgRescaleMultiplier: 0,
    cfgScale: settings.cfgScale,
    clipEmbedModel: components.clipEmbedModel,
    clipSkip: settings.clipSkip,
    componentSourceModel: null,
    loras: settings.loras,
    mistralEncoderModel: components.mistralEncoderModel,
    qwen35EncoderModel: components.qwen35EncoderModel,
    qwen3EncoderModel: components.qwen3EncoderModel,
    qwen3VLEncoderModel: components.qwen3VlEncoderModel,
    qwenVLEncoderModel: components.qwenVlEncoderModel,
    scheduler: coerceUpscaleScheduler(model, settings.scheduler),
    steps: settings.steps,
    t5EncoderModel: components.t5EncoderModel,
    vae: settings.vae,
    vaePrecision: settings.vaePrecision,
  };
};

/** Creativity's start, translated into the node's terms where it clips by step. */
const getUpscaleEntry = (
  model: MainModelConfig,
  denoiseSize: { width: number; height: number },
  { creativity, steps }: Pick<UpscaleWidgetValues, 'creativity' | 'steps'>
): number => {
  const valueStart = getUpscaleDenoisingStart(creativity);
  const clipping = getDenoisingStartClipping(model);

  return clipping?.kind === 'step'
    ? getIndexClippedDenoisingStart(valueStart, clipping.sigmas(model, denoiseSize, steps), clipping.rounding)
    : valueStart;
};

/** Structure's tile ControlNet: SD's two split across the schedule, or Z-Image's one, tiled like `i2l`. */
const addTileControl = (
  graph: BackendGraphContract,
  settings: UpscaleWidgetValues,
  kind: UpscaleTileControlNet['kind'],
  { denoise, frame }: { denoise: BackendInvocationContract; frame: BackendInvocationContract }
): void => {
  const { tileControlnetModel } = settings;

  if (!tileControlnetModel) {
    return;
  }

  if (kind === 'z_image_control') {
    const control = getZImageTileControlValues(settings.structure);
    const tileControl = addNode(graph, {
      begin_step_percent: control.beginStepPercent,
      control_context_scale: control.controlContextScale,
      control_model: toModelIdentifier(tileControlnetModel),
      end_step_percent: control.endStepPercent,
      id: 'tile_control',
      tile_size: settings.tileSize,
      tiled: true,
      type: 'z_image_control',
    });

    addEdge(graph, frame, 'image', tileControl, 'image');
    addEdge(graph, tileControl, 'control', denoise, 'control');

    return;
  }

  const control = getUpscaleControlNetValues(settings.structure);
  const collector = addNode(graph, { id: 'controlnet_collector', type: 'collect' });

  for (const [id, values] of [
    ['controlnet_1', control.first],
    ['controlnet_2', control.second],
  ] as const) {
    const controlNet = addNode(graph, {
      begin_step_percent: values.beginStepPercent,
      control_mode: 'balanced',
      control_model: toModelIdentifier(tileControlnetModel),
      control_weight: values.controlWeight,
      end_step_percent: values.endStepPercent,
      id,
      resize_mode: 'just_resize',
      type: 'controlnet',
    });

    addEdge(graph, frame, 'image', controlNet, 'image');
    addEdge(graph, controlNet, 'control', collector, 'item');
  }
  addEdge(graph, collector, 'collection', denoise, 'control');
};

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
  const architecture = upscaleArchitectureFor(model);

  if (!inputImage || !model || !upscaleModel || !architecture) {
    throw new Error('Upscale settings are incomplete.');
  }

  const grid = upscaleDenoiseGrid(model);
  const denoiseSize = {
    height: upscaleDenoiseDimension(inputImage.height, settings.scale, grid),
    width: upscaleDenoiseDimension(inputImage.width, settings.scale, grid),
  };
  const { denoise, graph, latents, output, vaeModel } = buildImageToImagePipeline({
    denoiseSize,
    denoisingStart: getUpscaleEntry(model, denoiseSize, settings),
    model,
    multiDiffusion: architecture.multiDiffusion
      ? { tileOverlap: settings.tileOverlap, tileSize: settings.tileSize }
      : null,
    output: { id: 'upscale_output', isIntermediate: destination === 'canvas' },
    settings: toPipelineSettings(settings, model),
    useCpuNoise: projectSettings.useCpuNoise,
    vaeTileSize: settings.tileSize,
  });

  if (!latents) {
    throw new Error(`Upscale cannot encode an image for ${model.name}.`);
  }

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

  const frame = fitFrameToDenoiseSize(graph, unsharpMask, denoiseSize);
  // An untiled encode of the full frame does not fit consumer VRAM; the pipeline tiles the decode.
  const encode = addNode(graph, {
    ...(latents.fp32 === undefined ? {} : { fp32: latents.fp32 }),
    id: 'i2l',
    tile_size: settings.tileSize,
    tiled: true,
    type: latents.encodeType,
  });

  addEdge(graph, frame, 'image', encode, 'image');
  addEdge(graph, latents.vae.node, latents.vae.field, encode, 'vae');
  addEdge(graph, encode, 'latents', denoise, 'latents');
  if (architecture.tileControlNet) {
    addTileControl(graph, settings, architecture.tileControlNet.kind, { denoise, frame });
  }
  addUpscaleMetadata(graph, { frame, output, vaeModel }, settings, randDevice);

  return {
    backendGraph: graph,
    // Names the pass in the queue, so "multi-diffusion" is not claimed where none runs.
    graph: toGraphContract(
      graph,
      architecture.multiDiffusion ? `${model.name} multi-diffusion upscale` : `${model.name} upscale`
    ),
    negativePromptNodeId: 'negative_prompt',
    outputNodeId: 'upscale_output',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};
