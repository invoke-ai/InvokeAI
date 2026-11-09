/**
 * Pure canvas compiler over uploaded inputs. Resize through the processing grid and restore the bbox; hosted
 * providers are asked for one of their own sizes, take the bbox composite for edits, and have their result restored
 * to the bbox; reject unsupported modes.
 */

import type { SupportedGenerateBase } from '@features/generation/core/baseGenerationPolicies';
import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';
import type { FamilyGraph, FamilyLatentSpace } from '@features/generation/core/familyGraph';
import type { ComponentModelConfig, GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import {
  getGenerationValidationReasons,
  hasEnabledKrea2EditReferences,
} from '@features/generation/core/baseGenerationPolicies';
import {
  describeExternalCanvasRefusal,
  isExternalGenerationMode,
  isExternalTextOnlyCanvasModel,
} from '@features/generation/core/externalProvider';
import { buildExternalGraph, getFamilyGraphBuilder } from '@features/generation/core/graph';
import {
  addEdge,
  addNode,
  addOutputPostprocessors,
  renameNode,
  replaceCanvasOutput,
  toGraphContract,
} from '@features/generation/core/graphBuilder';
import { addKrea2ConditioningEnhancers } from '@features/generation/core/krea2Conditioning';
import { getIsPidActive } from '@features/generation/core/pid';
import { shouldUsePidDecode } from '@features/generation/core/pidGraph';
import { getEffectivePrompts } from '@features/generation/core/promptTemplates';
import { getBaseDenoisingRange } from '@features/generation/core/sdxlRefiner';
import { MAX_ANIMA_INPAINT_ADAPTER_WEIGHT, MIN_ANIMA_INPAINT_ADAPTER_WEIGHT } from '@features/generation/core/settings';

import type { CanvasInpaintSources } from './addControlLayers';
import type {
  CanvasCompositingSettings,
  CanvasInfillMethod,
  CompileCanvasGraphInput,
  CompiledCanvasGraph,
  Rect,
} from './types';

import { addControlLayers, getOrAddLliteCollector } from './addControlLayers';
import {
  addRegionalGuidance,
  getEffectiveLayoutKind,
  getLayoutPromptRejectionReason,
  getLayoutRegionIssue,
  getRegionalGuidanceSupport,
  isLayoutRegionalGuidanceBase,
  isRegionalGuidanceSupportedForBase,
  type LayoutRegionalGuidanceBase,
  type RegionalGuidanceInput,
  type RegionalLayoutKind,
} from './addRegionalGuidance';
import { type CanvasSize, resolveCanvasProcessingSize } from './canvasProcessingSize';
import { isAnimaInpaintAdapterModel } from './controlValidation';

/**
 * SD-3, FLUX, FLUX.2, FLUX.3 and MOSS-Image use start = 1 - strength^0.2; MOSS-Image clips its shifted schedule in
 * sigma space like FLUX.2. FLUX Fill always starts at 0; other bases use 1 - strength.
 */
const canvasDenoisingStart = (model: GenerateModelConfig, strength: number): number => {
  if (model.base === 'flux' && model.variant === 'dev_fill') {
    return 0;
  }
  const usesOptimizedCurve =
    model.base === 'sd-3' ||
    model.base === 'flux' ||
    model.base === 'flux2' ||
    model.base === 'flux3' ||
    model.base === 'moss-image';
  return 1 - strength ** (usesOptimizedCurve ? 0.2 : 1);
};

/** Re-derives the base denoise range from the strength; the refiner hand-over rule stays with its owner. */
const setCanvasDenoisingRange = (
  denoise: BackendInvocationContract,
  model: GenerateModelConfig,
  settings: GenerateSettings,
  strength: number
): void => {
  Object.assign(denoise, getBaseDenoisingRange(settings, model, canvasDenoisingStart(model, strength)));
};

/** True for the image-referencing modes (everything but pure txt2img). */
const isImageMode = (mode: CompileCanvasGraphInput['mode']): boolean => mode !== 'txt2img';

const sizesMatch = (left: CanvasSize, right: CanvasSize): boolean =>
  left.width === right.width && left.height === right.height;

/** Whether an Anima inpaint/outpaint graph gets the adapter: it is switched on and picked, and the base is Anima. */
const hasAnimaInpaintAdapter = (model: GenerateModelConfig, settings: GenerateSettings): boolean =>
  model.base === 'anima' && settings.animaInpaintAdapterEnabled && settings.animaInpaintAdapterModel !== null;

/**
 * The legacy PiD guards (`buildSDXLGraph.ts:192–205`, repeated in the other builders): only the txt2img and img2img
 * graphs decode through PiD, and Scale Before Processing must leave the bbox alone because PiD already decodes at
 * 4× the generation size.
 */
const getCanvasPidValidationReasons = (input: CompileCanvasGraphInput, processingSize: CanvasSize): string[] => {
  const { bbox, mode, model, settings } = input;
  const reasons: string[] = [];

  if (mode === 'inpaint' || mode === 'outpaint') {
    reasons.push(
      'PiD decode currently supports Text to Image and Image to Image only. Disable PiD for Inpaint/Outpaint.'
    );
  }

  const unscaledSize = resolveCanvasProcessingSize(model, settings.pidMode, bbox, undefined);
  if (!sizesMatch(processingSize, unscaledSize)) {
    reasons.push('Turn off Scale Before Processing (set it to None) to use PiD decode.');
  }

  return reasons;
};

/** Canvas-specific validation reasons layered on top of the shared generate ones. */
const getCanvasValidationReasons = (input: CompileCanvasGraphInput, processingSize: CanvasSize): string[] => {
  const { bbox, compositeImageName, maskImageName, mode, model, settings, strength } = input;
  const reasons: string[] = [];

  if (model.type === 'external_image_generator') {
    if (mode !== 'txt2img' && isExternalTextOnlyCanvasModel(model)) {
      // Every edit mode comes from raster content in the frame, which names the cause better than the mode does.
      reasons.push(describeExternalCanvasRefusal(model, { kind: 'raster_content' }));
    } else if (mode === 'inpaint') {
      // The Canvas keep-mask does not match provider edit-mask contracts, and there is no masked compositing tail.
      reasons.push(`${model.name} does not support inpainting on the canvas yet. Remove or disable the inpaint mask.`);
    } else if (mode === 'outpaint') {
      reasons.push(`${model.name} does not support outpainting on the canvas.`);
    }
    if (input.controlLayers && input.controlLayers.length > 0) {
      reasons.push(`${model.name} does not support control layers.`);
    }
    if (input.regionalGuidance && input.regionalGuidance.length > 0) {
      reasons.push(`${model.name} does not support regional guidance.`);
    }
  }

  if (getIsPidActive(settings.pidMode, model.base)) {
    reasons.push(...getCanvasPidValidationReasons(input, processingSize));
  }

  if ((mode === 'inpaint' || mode === 'outpaint') && hasAnimaInpaintAdapter(model, settings)) {
    if (!isAnimaInpaintAdapterModel(settings.animaInpaintAdapterModel as ComponentModelConfig)) {
      reasons.push('The Anima inpaint adapter must be a 4-channel Anima ControlNet-LLLite model.');
    }
    const weight = settings.animaInpaintAdapterWeight;
    if (
      !Number.isFinite(weight) ||
      weight < MIN_ANIMA_INPAINT_ADAPTER_WEIGHT ||
      weight > MAX_ANIMA_INPAINT_ADAPTER_WEIGHT
    ) {
      reasons.push(
        `Anima inpaint adapter weight must be between ${MIN_ANIMA_INPAINT_ADAPTER_WEIGHT} and ${MAX_ANIMA_INPAINT_ADAPTER_WEIGHT}.`
      );
    }
  }

  // A regional mask covers the prompt and target tokens only; edit references add image tokens it cannot place.
  if (
    model.base === 'krea-2' &&
    (input.regionalGuidance?.length ?? 0) > 0 &&
    hasEnabledKrea2EditReferences(settings.referenceImages)
  ) {
    reasons.push(
      'Krea-2 edit references cannot be combined with regional guidance. Turn off the edit references or disable the regional guidance layers.'
    );
  }

  if (!Number.isFinite(bbox.width) || !Number.isFinite(bbox.height) || bbox.width <= 0 || bbox.height <= 0) {
    reasons.push('Canvas frame must have a positive area.');
  } else if (!Number.isInteger(bbox.width) || !Number.isInteger(bbox.height)) {
    reasons.push('Canvas frame dimensions must be whole pixels.');
  }

  if (isImageMode(mode)) {
    if (!compositeImageName) {
      reasons.push('Canvas generation requires a composited source image.');
    }

    // A provider edits at its own strength; only the local denoise reads this one.
    if (model.type !== 'external_image_generator' && (!Number.isFinite(strength) || strength <= 0 || strength > 1)) {
      reasons.push('Canvas denoising strength must be greater than 0 and at most 1.');
    }
  }

  // Inpaint always needs a mask (an active inpaint mask defines the region);
  // outpaint derives its mask from the raster alpha, so its mask is optional.
  if (mode === 'inpaint' && !maskImageName) {
    reasons.push('Canvas inpainting requires an inpaint mask.');
  }

  // A layout is built around the global prompt, so regions around a blank one are refused. A mask with pixels in the
  // bbox has an area to place.
  const layoutPromptReason = getLayoutPromptRejectionReason({
    base: model.base,
    prompt: getEffectivePrompts(settings).positivePrompt,
    regions: (input.regionalGuidance ?? []).filter(
      (region) => region.maskBounds.width > 0 && region.maskBounds.height > 0
    ),
  });
  if (layoutPromptReason) {
    reasons.push(layoutPromptReason);
  }

  return reasons;
};

/** The settings a base builder sees: identical to the widget's, but sized to the model-valid processing frame. */
const withProcessingDimensions = (settings: GenerateSettings, processingSize: CanvasSize): GenerateSettings => ({
  ...settings,
  height: processingSize.height,
  width: processingSize.width,
});

const findMetadataNode = (graph: BackendGraphContract): BackendInvocationContract | undefined =>
  Object.values(graph.nodes).find((node) => node.type === 'core_metadata');

/** Resizes a model-grid processing output back to the bbox's exact final footprint. */
const resizeCanvasOutputToBbox = (
  graph: BackendGraphContract,
  bbox: CompileCanvasGraphInput['bbox'],
  processingSize: CanvasSize,
  destination: CompileCanvasGraphInput['destination'],
  /** PiD Fit ends in an `img_resize` of the 4x decode to the processing size (`addPidDecode`). */
  isPidFit: boolean
): void => {
  if (sizesMatch(bbox, processingSize)) {
    return;
  }

  // Retarget PiD Fit's resize so the output is resampled once, straight to the bbox (legacy parity), rather
  // than through the processing size.
  const terminalResize = graph.nodes.canvas_output;
  if (isPidFit && terminalResize?.type === 'img_resize') {
    terminalResize.width = bbox.width;
    terminalResize.height = bbox.height;
    return;
  }

  const { output, retired } = replaceCanvasOutput(graph, 'canvas_processing_output', {
    height: bbox.height,
    is_intermediate: destination === 'canvas',
    type: 'img_resize',
    use_cache: false,
    width: bbox.width,
  });
  addEdge(graph, retired, 'image', output, 'image');
};

/**
 * Hosted nodes return an image collection; restore each image to the bbox with its own provider metadata. A
 * text-to-image result at another aspect ratio is scaled to cover the bbox and center-cropped, never stretched.
 * Scaling before cropping keeps the crop centered when a provider answers at another scale of the requested shape
 * (named sizes such as Gemini's `2K`). An edit result is resized plainly, undoing its init image's resize.
 */
const restoreExternalCanvasOutputToBbox = (
  graph: BackendGraphContract,
  input: Pick<CompileCanvasGraphInput, 'bbox' | 'destination'> & { mode: 'txt2img' | 'img2img' },
  requestSize: CanvasSize
): void => {
  const { bbox, destination, mode } = input;
  const provider = renameNode(graph, 'canvas_output', 'canvas_processing_output');
  provider.is_intermediate = true;
  const image = addNode(graph, { id: 'canvas_provider_image', type: 'iterate', use_cache: false });
  const metadata = addNode(graph, { id: 'canvas_provider_metadata', type: 'metadata_from_image' });
  const output = { id: 'canvas_output', is_intermediate: destination === 'canvas', use_cache: false };
  // Keep core metadata on the provider so its output adds request identity, defaults and actual seed to it.
  addEdge(graph, provider, 'collection', image, 'collection');
  addEdge(graph, image, 'item', metadata, 'image');

  if (mode === 'img2img' || requestSize.width * bbox.height === requestSize.height * bbox.width) {
    const resize = addNode(graph, { ...output, height: bbox.height, type: 'img_resize', width: bbox.width });
    addEdge(graph, image, 'item', resize, 'image');
    addEdge(graph, metadata, 'metadata', resize, 'metadata');
    return;
  }

  const scale = Math.max(bbox.width / requestSize.width, bbox.height / requestSize.height);
  const cover = addNode(graph, {
    height: Math.max(bbox.height, Math.round(requestSize.height * scale)),
    id: 'canvas_provider_cover',
    is_intermediate: true,
    type: 'img_resize',
    use_cache: false,
    width: Math.max(bbox.width, Math.round(requestSize.width * scale)),
  });
  const crop = addNode(graph, {
    ...output,
    height: bbox.height,
    type: 'img_crop',
    width: bbox.width,
    x: Math.floor((cover.width - bbox.width) / 2),
    y: Math.floor((cover.height - bbox.height) / 2),
  });
  addEdge(graph, image, 'item', cover, 'image');
  addEdge(graph, cover, 'image', crop, 'image');
  addEdge(graph, metadata, 'metadata', crop, 'metadata');
};

/** Feeds the provider the composite at the request size, as the backend would have resized it to its own bucket. */
const resizeExternalInitImage = (graph: BackendGraphContract, imageName: string, requestSize: CanvasSize): void => {
  const provider = graph.nodes.canvas_output as BackendInvocationContract;
  delete provider.init_image;
  addEdge(
    graph,
    addInputResizeNode(graph, 'canvas_resize_image_to_request', imageName, requestSize),
    'image',
    provider,
    'init_image'
  );
};

/** Keeps saved-image metadata aligned to the bbox footprint, not the internal processing size. */
const setCanvasMetadataDimensions = (graph: BackendGraphContract, bbox: CompileCanvasGraphInput['bbox']): void => {
  const metadata = findMetadataNode(graph);
  if (metadata) {
    metadata.width = bbox.width;
    metadata.height = bbox.height;
  }
};

/** Record only the resolved inputs handed to the graph, beside the independently versioned authored document. */
const recordCanvasInputs = (graph: BackendGraphContract, input: CompileCanvasGraphInput): void => {
  const metadata = findMetadataNode(graph);
  if (!metadata) {
    return;
  }
  if (input.canvasMetadata) {
    metadata.canvas_v3_metadata = input.canvasMetadata;
  }
  const controls = input.controlLayers ?? [];
  const recordedControls = controls.map((control) => ({
    begin_step_percent: control.beginEndStepPct[0],
    end_step_percent: control.beginEndStepPct[1],
    control_mode: control.controlMode,
    image: { image_name: control.imageName, width: input.bbox.width, height: input.bbox.height },
    position: { x: input.bbox.x, y: input.bbox.y },
    layer_id: control.id,
    model: control.model,
    type: control.kind,
    weight: control.weight,
    ...(control.instantxControlMode === null || control.instantxControlMode === undefined
      ? {}
      : { instantx_control_mode: control.instantxControlMode }),
  }));
  metadata.control_layers = recordedControls;
  metadata.controlnets = recordedControls
    .filter((control) => control.type === 'controlnet')
    .map((control) => ({
      image: control.image,
      control_model: control.model,
      control_weight: control.weight,
      begin_step_percent: control.begin_step_percent,
      end_step_percent: control.end_step_percent,
      control_mode: control.control_mode ?? 'balanced',
    }));
  metadata.t2iAdapters = recordedControls
    .filter((control) => control.type === 't2i_adapter')
    .map((control) => ({
      image: control.image,
      t2i_adapter_model: control.model,
      weight: control.weight,
      begin_step_percent: control.begin_step_percent,
      end_step_percent: control.end_step_percent,
    }));
  const layoutRegions = graph.nodes.layout_prompt?.regions ?? graph.nodes.caption_builder?.regions;
  if (Array.isArray(layoutRegions) && layoutRegions.length > 0) {
    metadata.regions = layoutRegions;
  } else if (!['moss-image', 'ideogram-4', 'flux3'].includes(input.model.base)) {
    const regions = (input.regionalGuidance ?? []).filter((region) => graph.nodes[`rg_mask_to_tensor_${region.id}`]);
    if (regions.length) {
      metadata.regions = regions.map(({ layout: _layout, ...region }) => region);
    }
  }
};

/** Sets the metadata `generation_mode` variant + strength (legacy parity). */
const setMetadataMode = (graph: BackendGraphContract, mode: string, strength: number): void => {
  const metadata = findMetadataNode(graph);
  if (metadata) {
    if (typeof metadata.generation_mode === 'string') {
      metadata.generation_mode = metadata.generation_mode.replace('txt2img', mode);
    }
    metadata.strength = strength;
  }
};

/** Adds the family's image-to-latents encode node fed by `imageName`. */
const addEncodeNode = (
  graph: BackendGraphContract,
  latents: FamilyLatentSpace,
  settings: GenerateSettings,
  imageName: string | null
): BackendInvocationContract =>
  addNode(graph, {
    id: 'canvas_i2l',
    ...(imageName ? { image: { image_name: imageName } } : {}),
    type: latents.encodeType,
    ...(latents.fp32 === undefined ? {} : { fp32: latents.fp32 }),
    ...(latents.encodeType === 'i2l' ? { color_compensation: settings.colorCompensation ? 'SDXL' : 'None' } : {}),
  });

/** Encodes with the VAE the family decodes with and feeds the latents to its denoise. */
const connectEncode = (family: FamilyGraph, latents: FamilyLatentSpace, encode: BackendInvocationContract): void => {
  addEdge(family.graph, latents.vae.node, latents.vae.field, encode, 'vae');
  addEdge(family.graph, encode, 'latents', family.denoise, 'latents');
};

const addInputResizeNode = (
  graph: BackendGraphContract,
  id: string,
  imageName: string,
  processingSize: CanvasSize
): BackendInvocationContract =>
  addNode(graph, {
    height: processingSize.height,
    id,
    image: { image_name: imageName },
    is_intermediate: true,
    type: 'img_resize',
    width: processingSize.width,
  });

/** The family's latent space, for the image modes that encode into it (throws when the family has none). */
const requireLatents = (family: FamilyGraph, model: GenerateModelConfig): FamilyLatentSpace => {
  if (!family.latents) {
    throw new Error(`Canvas generation is not supported for ${model.name}.`);
  }
  return family.latents;
};

/** Grafts the image-to-latents encode path onto a freshly built base graph (img2img). */
const graftImageToImage = (
  family: FamilyGraph,
  model: GenerateModelConfig,
  settings: GenerateSettings,
  compositeImageName: string,
  strength: number,
  bbox: CompileCanvasGraphInput['bbox']
): void => {
  const { graph } = family;
  const latents = requireLatents(family, model);
  // The source image is encoded at the size the graph denoises at. That is the processing size, except under
  // native PiD, which denoises at a quarter of it and decodes straight back up (legacy `addPidImageToImageNative`).
  const encodeSize = family.denoiseSize;
  const needsResize = !sizesMatch(bbox, encodeSize);
  const encode = addEncodeNode(graph, latents, settings, needsResize ? null : compositeImageName);

  if (needsResize) {
    const resize = addInputResizeNode(graph, 'canvas_resize_initial_to_processing', compositeImageName, encodeSize);
    addEdge(graph, resize, 'image', encode, 'image');
  }

  connectEncode(family, latents, encode);
  setCanvasDenoisingRange(family.denoise, model, settings, strength);

  setMetadataMode(graph, 'img2img', strength);
};

/** The backend infill node for the selected method (legacy `getInfill`). */
const addInfillNode = (
  graph: BackendGraphContract,
  method: CanvasInfillMethod,
  compositing: CanvasCompositingSettings
): BackendInvocationContract => {
  switch (method) {
    case 'patchmatch':
      return addNode(graph, {
        downscale: compositing.infillPatchmatchDownscaleSize,
        id: 'infill',
        type: 'infill_patchmatch',
      });
    case 'lama':
      return addNode(graph, { id: 'infill', type: 'infill_lama' });
    case 'cv2':
      return addNode(graph, { id: 'infill', type: 'infill_cv2' });
    case 'tile':
      return addNode(graph, { id: 'infill', tile_size: compositing.infillTileSize, type: 'infill_tile' });
    case 'color': {
      const { a, b, g, r } = compositing.infillColorValue;
      return addNode(graph, {
        color: { a: Math.round(a * 255), b, g, r },
        id: 'infill',
        type: 'infill_rgba',
      });
    }
  }
};

/**
 * Anima's 4-channel ControlNet-LLLite inpaint adapter: it conditions on the raster image and the region to inpaint
 * (legacy `addInpaint.ts` / `addOutpaint.ts`). The canvas mask is white = keep, the adapter expects white =
 * inpaint, so `img_lerp` with `min: 255, max: 0` inverts it first. The adapter joins the control layers' collector;
 * `anima_denoise` applies each LLLite model once, and control layers admit only 3-channel models, so they never
 * collide.
 */
const addAnimaInpaintAdapter = (
  graph: BackendGraphContract,
  args: {
    denoise: BackendInvocationContract;
    inpaint: CanvasInpaintSources;
    model: ComponentModelConfig;
    weight: number;
  }
): void => {
  const { denoise, inpaint, model, weight } = args;
  const { image, mask } = inpaint;
  const invertMask = addNode(graph, {
    id: 'invert_anima_inpaint_adapter_mask',
    ...('image_name' in mask ? { image: mask } : {}),
    max: 0,
    min: 255,
    type: 'img_lerp',
  });
  if ('node' in mask) {
    addEdge(graph, mask.node, mask.field, invertMask, 'image');
  }

  const adapter = addNode(graph, {
    control_model: model,
    id: 'anima_inpaint_adapter',
    ...('image_name' in image ? { image } : {}),
    type: 'anima_lllite',
    weight,
  });
  if ('node' in image) {
    addEdge(graph, image.node, image.field, adapter, 'image');
  }
  addEdge(graph, invertMask, 'image', adapter, 'mask');
  addEdge(graph, adapter, 'control', getOrAddLliteCollector(graph, denoise), 'item');

  const metadata = findMetadataNode(graph);
  if (metadata) {
    metadata.anima_inpaint_adapter = model;
    metadata.anima_inpaint_adapter_weight = weight;
  }
};

/**
 * outputOnlyMaskedRegions selects transparent output versus compositing with the original. Returns what the
 * inpaint-conditioned adapters read.
 */
const graftMaskTail = (
  family: FamilyGraph,
  args: {
    model: GenerateModelConfig;
    settings: GenerateSettings;
    latents: FamilyLatentSpace;
    initialImageName: string;
    compositing: CanvasCompositingSettings;
    destination: CompileCanvasGraphInput['destination'];
    bbox: CompileCanvasGraphInput['bbox'];
    processingSize: CanvasSize;
    gradientImageEdge?: { node: BackendInvocationContract; field: string };
    /** Either a fixed mask image field or an edge-fed source node. */
    gradientMaskImage?: { image_name: string };
    gradientMaskEdge?: { node: BackendInvocationContract; field: string };
    /** The raster image the inpaint-conditioned adapters read; the composite itself when absent. */
    adapterImageEdge?: { node: BackendInvocationContract; field: string };
  }
): CanvasInpaintSources => {
  const { compositing, destination, initialImageName, latents, model, settings } = args;
  const { denoise, graph } = family;
  const needsResize = !sizesMatch(args.bbox, args.processingSize);

  // The base decoder, until the composited output below takes over canvas_output from it.
  const l2i = graph.nodes.canvas_output;
  if (!l2i) {
    throw new Error('Canvas generation could not find the base graph output.');
  }

  const gradientMask = addNode(graph, {
    coherence_mode: compositing.coherenceMode,
    edge_radius: compositing.coherenceEdgeSize,
    fp32: latents.fp32 ?? false,
    id: 'create_gradient_mask',
    ...(args.gradientImageEdge ? {} : { image: { image_name: initialImageName } }),
    minimum_denoise: compositing.coherenceMinDenoise,
    type: 'create_gradient_mask',
    ...(args.gradientMaskImage ? { mask: args.gradientMaskImage } : {}),
  });

  if (args.gradientMaskEdge) {
    addEdge(graph, args.gradientMaskEdge.node, args.gradientMaskEdge.field, gradientMask, 'mask');
  }
  if (args.gradientImageEdge) {
    addEdge(graph, args.gradientImageEdge.node, args.gradientImageEdge.field, gradientMask, 'image');
  }
  addEdge(graph, latents.vae.node, latents.vae.field, gradientMask, 'vae');
  // The optional UNet edge only applies to SD-family models (legacy `isMainModelWithoutUnet`).
  if (latents.unet) {
    addEdge(graph, latents.unet.node, latents.unet.field, gradientMask, 'unet');
  }
  addEdge(graph, gradientMask, 'denoise_mask', denoise, 'denoise_mask');

  // The image before any noise is added, and the mask the denoise limits itself to, both at processing size.
  const inpaint: CanvasInpaintSources = {
    image: args.adapterImageEdge ?? { image_name: initialImageName },
    mask: args.gradientMaskEdge ?? (args.gradientMaskImage as { image_name: string }),
  };
  if (hasAnimaInpaintAdapter(model, settings)) {
    addAnimaInpaintAdapter(graph, {
      denoise,
      inpaint,
      model: settings.animaInpaintAdapterModel as ComponentModelConfig,
      weight: settings.animaInpaintAdapterWeight,
    });
  }

  const expandMask = addNode(graph, {
    fade_size_px: compositing.maskBlur,
    id: 'expand_mask',
    type: 'expand_mask_with_fade',
  });
  addEdge(graph, gradientMask, 'expanded_mask_area', expandMask, 'mask');

  let generatedImageSource: BackendInvocationContract = l2i;
  let outputMaskSource: BackendInvocationContract = expandMask;

  if (needsResize) {
    generatedImageSource = addNode(graph, {
      height: args.bbox.height,
      id: 'canvas_resize_generated_to_bbox',
      is_intermediate: true,
      type: 'img_resize',
      width: args.bbox.width,
    });
    outputMaskSource = addNode(graph, {
      height: args.bbox.height,
      id: 'canvas_resize_output_mask_to_bbox',
      is_intermediate: true,
      type: 'img_resize',
      width: args.bbox.width,
    });
    addEdge(graph, l2i, 'image', generatedImageSource, 'image');
    addEdge(graph, expandMask, 'image', outputMaskSource, 'image');
  }

  const { output } = replaceCanvasOutput(
    graph,
    'canvas_l2i',
    compositing.outputOnlyMaskedRegions
      ? {
          invert_mask: true,
          is_intermediate: destination === 'canvas',
          type: 'apply_mask_to_image',
          use_cache: false,
        }
      : {
          is_intermediate: destination === 'canvas',
          layer_base: { image_name: initialImageName },
          type: 'invokeai_img_blend',
          use_cache: false,
        }
  );
  addEdge(graph, generatedImageSource, 'image', output, compositing.outputOnlyMaskedRegions ? 'image' : 'layer_upper');
  addEdge(graph, outputMaskSource, 'image', output, 'mask');
  return inpaint;
};

/** Optionally inserts an `img_noise` node before encode; returns the node feeding `i2l.image`. */
const addNoiseBeforeEncode = (
  graph: BackendGraphContract,
  imageName: string,
  noiseMaskImageName: string | null | undefined,
  imageSourceNode: BackendInvocationContract | null,
  noiseMaskSourceNode: BackendInvocationContract | null = null
): { imageField?: { image_name: string }; edgeFrom?: BackendInvocationContract } => {
  if (!noiseMaskImageName) {
    // No noise mask: encode reads the (infilled) initial image directly.
    return imageSourceNode ? { edgeFrom: imageSourceNode } : { imageField: { image_name: imageName } };
  }
  const noise = addNode(graph, {
    amount: 1.0,
    id: 'add_inpaint_noise',
    ...(imageSourceNode ? {} : { image: { image_name: imageName } }),
    ...(noiseMaskSourceNode ? {} : { mask: { image_name: noiseMaskImageName } }),
    noise_color: true,
    noise_type: 'gaussian',
    type: 'img_noise',
  });
  addEdge(graph, graph.nodes.seed, 'value', noise, 'seed');
  if (imageSourceNode) {
    addEdge(graph, imageSourceNode, 'image', noise, 'image');
  }
  if (noiseMaskSourceNode) {
    addEdge(graph, noiseMaskSourceNode, 'image', noise, 'mask');
  }
  return { edgeFrom: noise };
};

/** Grafts the inpaint pipeline (content covers the bbox, an inpaint mask restricts it). */
const graftInpaint = (
  family: FamilyGraph,
  input: CompileCanvasGraphInput,
  compositing: CanvasCompositingSettings,
  processingSize: CanvasSize
): CanvasInpaintSources => {
  const { model } = input;
  const { denoise, graph } = family;
  const latents = requireLatents(family, model);
  const initialImageName = input.compositeImageName as string;
  const maskImageName = input.maskImageName as string;
  const strength = input.strength;
  const needsResize = !sizesMatch(input.bbox, processingSize);

  setCanvasDenoisingRange(denoise, model, input.settings, strength);

  const initialResize = needsResize
    ? addInputResizeNode(graph, 'canvas_resize_initial_to_processing', initialImageName, processingSize)
    : null;
  const maskResize = needsResize
    ? addInputResizeNode(graph, 'canvas_resize_mask_to_processing', maskImageName, processingSize)
    : null;
  const noiseMaskResize =
    needsResize && input.noiseMaskImageName
      ? addInputResizeNode(graph, 'canvas_resize_noise_mask_to_processing', input.noiseMaskImageName, processingSize)
      : null;

  const encode = addEncodeNode(graph, latents, input.settings, null);
  const noiseResult = addNoiseBeforeEncode(
    graph,
    initialImageName,
    input.noiseMaskImageName,
    initialResize,
    noiseMaskResize
  );
  if (noiseResult.edgeFrom) {
    addEdge(graph, noiseResult.edgeFrom, 'image', encode, 'image');
  } else if (noiseResult.imageField) {
    encode.image = noiseResult.imageField;
  }
  connectEncode(family, latents, encode);

  const inpaint = graftMaskTail(family, {
    ...(initialResize ? { adapterImageEdge: { field: 'image', node: initialResize } } : {}),
    bbox: input.bbox,
    compositing,
    destination: input.destination,
    ...(initialResize ? { gradientImageEdge: { field: 'image', node: initialResize } } : {}),
    ...(maskResize
      ? { gradientMaskEdge: { field: 'image', node: maskResize } }
      : { gradientMaskImage: { image_name: maskImageName } }),
    initialImageName,
    latents,
    model,
    processingSize,
    settings: input.settings,
  });

  setMetadataMode(graph, 'inpaint', strength);
  return inpaint;
};

/** Grafts the outpaint pipeline (bbox extends past content / has transparent holes). */
const graftOutpaint = (
  family: FamilyGraph,
  input: CompileCanvasGraphInput,
  compositing: CanvasCompositingSettings,
  processingSize: CanvasSize
): CanvasInpaintSources => {
  const { model } = input;
  const { denoise, graph } = family;
  const latents = requireLatents(family, model);
  const initialImageName = input.compositeImageName as string;
  const strength = input.strength;
  const needsResize = !sizesMatch(input.bbox, processingSize);

  setCanvasDenoisingRange(denoise, model, input.settings, strength);

  // Infill the transparent region before encode (legacy `getInfill`).
  const infill = addInfillNode(graph, compositing.infillMethod, compositing);
  const initialResize = needsResize
    ? addInputResizeNode(graph, 'canvas_resize_initial_to_processing', initialImageName, processingSize)
    : null;
  if (initialResize) {
    addEdge(graph, initialResize, 'image', infill, 'image');
  } else {
    infill.image = { image_name: initialImageName };
  }

  // Derive the mask from image alpha (transparent means generate), combining an explicit inpaint mask when
  // present.
  const alphaToMask = addNode(graph, {
    id: 'image_alpha_to_mask',
    image: { image_name: initialImageName },
    type: 'tomask',
  });

  let gradientMaskEdge: { node: BackendInvocationContract; field: string };
  if (input.maskImageName) {
    const maskCombine = addNode(graph, {
      id: 'mask_combine',
      mask1: { image_name: input.maskImageName },
      type: 'mask_combine',
    });
    addEdge(graph, alphaToMask, 'image', maskCombine, 'mask2');
    gradientMaskEdge = { field: 'image', node: maskCombine };
  } else {
    gradientMaskEdge = { field: 'image', node: alphaToMask };
  }

  if (needsResize) {
    const maskResize = addNode(graph, {
      height: processingSize.height,
      id: 'canvas_resize_outpaint_mask_to_processing',
      is_intermediate: true,
      type: 'img_resize',
      width: processingSize.width,
    });
    addEdge(graph, gradientMaskEdge.node, gradientMaskEdge.field, maskResize, 'image');
    gradientMaskEdge = { field: 'image', node: maskResize };
  }

  const noiseMaskResize =
    needsResize && input.noiseMaskImageName
      ? addInputResizeNode(graph, 'canvas_resize_noise_mask_to_processing', input.noiseMaskImageName, processingSize)
      : null;

  const encode = addEncodeNode(graph, latents, input.settings, null);
  const noiseResult = addNoiseBeforeEncode(graph, initialImageName, input.noiseMaskImageName, infill, noiseMaskResize);
  if (noiseResult.edgeFrom) {
    addEdge(graph, noiseResult.edgeFrom, 'image', encode, 'image');
  }
  connectEncode(family, latents, encode);

  const inpaint = graftMaskTail(family, {
    adapterImageEdge: { field: 'image', node: infill },
    bbox: input.bbox,
    compositing,
    destination: input.destination,
    ...(needsResize ? { gradientImageEdge: { field: 'image', node: infill } } : {}),
    gradientMaskEdge,
    initialImageName,
    latents,
    model,
    processingSize,
    settings: input.settings,
  });

  setMetadataMode(graph, 'outpaint', strength);
  return inpaint;
};

/** One element of the `moss_image_layout_prompt` node's `regions` input; `bbox` is `[x1, y1, x2, y2]` pixels. */
interface LayoutRegionElement {
  kind: RegionalLayoutKind;
  caption: string;
  /** Present exactly for `text` elements, as the node requires. */
  text?: string;
  bbox: [number, number, number, number];
}

interface AddLayoutRegionsOptions {
  /** The generation bbox the regions' mask bounds are measured in. */
  bbox: Pick<Rect, 'width' | 'height'>;
  /** The roles the base distinguishes. */
  layoutKinds: readonly RegionalLayoutKind[];
  regions: readonly RegionalGuidanceInput[];
}

// Scaling a box that ends on the bbox edge can overshoot the image by a rounding error; the node refuses that.
const clamp = (value: number, max: number): number => Math.min(Math.max(value, 0), max);

const toLayoutElement = (
  region: RegionalGuidanceInput,
  layoutKinds: readonly RegionalLayoutKind[],
  scale: { x: number; y: number },
  size: { width: number; height: number }
): LayoutRegionElement | null => {
  const caption = region.positivePrompt;
  if (caption === null || getLayoutRegionIssue(region, layoutKinds) !== null) {
    return null;
  }
  const bounds = region.maskBounds;
  const x1 = clamp(bounds.x * scale.x, size.width);
  const y1 = clamp(bounds.y * scale.y, size.height);
  const x2 = clamp((bounds.x + bounds.width) * scale.x, size.width);
  const y2 = clamp((bounds.y + bounds.height) * scale.y, size.height);
  // A mask that misses the bbox has no area there; the node refuses empty boxes.
  if (x2 <= x1 || y2 <= y1) {
    return null;
  }
  const kind = getEffectiveLayoutKind(layoutKinds, region.layout);
  return {
    bbox: [x1, y1, x2, y2],
    caption,
    kind,
    ...(kind === 'text' ? { text: region.layout?.text } : {}),
  };
};

/**
 * MOSS-Image regions: each becomes a box around its mask, scaled from bbox pixels to the size the denoiser generates
 * (which PiD can shrink below the processing size) and clamped to it. The base graph's `layout_prompt` node assembles
 * them with the global prompt at run time; regions the layout cannot place are left out.
 */
const addMossImageLayoutRegions = (family: FamilyGraph, options: AddLayoutRegionsOptions): void => {
  const { layoutPrompt } = family;
  if (!layoutPrompt) {
    throw new Error('addMossImageLayoutRegions: base graph is missing the layout prompt node.');
  }
  const size = family.denoiseSize;
  const scale = { x: size.width / options.bbox.width, y: size.height / options.bbox.height };
  // The frame and the boxes come from one size, so they cannot drift apart.
  layoutPrompt.width = size.width;
  layoutPrompt.height = size.height;
  layoutPrompt.regions = options.regions.flatMap(
    (region) => toLayoutElement(region, options.layoutKinds, scale, size) ?? []
  );
};

/** A box `[y_min, x_min, y_max, x_max]` on the grid Ideogram 4 and FLUX.3 share. */
type GridBox = [number, number, number, number];

/** Grid boxes are whole numbers from 0 to 1000 across the image, origin top-left. */
const GRID_COORD_MAX = 1000;

const toGridCoord = (pixels: number, extent: number): number =>
  clamp(Math.round((pixels / extent) * GRID_COORD_MAX), GRID_COORD_MAX);

/** A mask's bounds on the grid over the bbox, or null when the mask misses the bbox or rounds to a line. */
const toGridBox = (bounds: Rect, bbox: Pick<Rect, 'width' | 'height'>): GridBox | null => {
  const yMin = toGridCoord(bounds.y, bbox.height);
  const xMin = toGridCoord(bounds.x, bbox.width);
  const yMax = toGridCoord(bounds.y + bounds.height, bbox.height);
  const xMax = toGridCoord(bounds.x + bounds.width, bbox.width);
  return yMax > yMin && xMax > xMin ? [yMin, xMin, yMax, xMax] : null;
};

/** One element of the `ideogram4_caption_builder` node's `regions` input. */
interface Ideogram4RegionElement {
  prompt: string;
  bbox: GridBox;
}

const toIdeogram4Region = (
  region: RegionalGuidanceInput,
  options: AddLayoutRegionsOptions
): Ideogram4RegionElement | null => {
  const prompt = region.positivePrompt;
  if (prompt === null || getLayoutRegionIssue(region, options.layoutKinds) !== null) {
    return null;
  }
  const bbox = toGridBox(region.maskBounds, options.bbox);
  return bbox ? { bbox, prompt } : null;
};

/**
 * Ideogram 4 regions: each becomes an object element of the structured caption, boxed around its mask. The output is
 * the bbox, resized, so boxes are normalized to the bbox rather than the denoise size. The base graph's
 * `caption_builder` assembles the caption around the global prompt at run time, so batch prompt variation still flows
 * through `positive_prompt`, and core metadata already records its output. Regions the caption cannot place are left
 * out, and without any the graph stays untouched.
 */
const addIdeogram4Regions = (family: FamilyGraph, options: AddLayoutRegionsOptions): void => {
  const captionBuilder = family.layoutPrompt;
  if (!captionBuilder) {
    throw new Error('addIdeogram4Regions: base graph is missing the caption builder node.');
  }
  const regions = options.regions.flatMap((region) => toIdeogram4Region(region, options) ?? []);
  if (regions.length > 0) {
    captionBuilder.regions = regions;
  }
};

/** One element of the `flux3_layout_prompt` node's `regions` input. */
interface Flux3RegionElement {
  desc: string;
  bbox: GridBox;
}

const toFlux3Region = (region: RegionalGuidanceInput, options: AddLayoutRegionsOptions): Flux3RegionElement | null => {
  const desc = region.positivePrompt;
  if (desc === null || getLayoutRegionIssue(region, options.layoutKinds) !== null) {
    return null;
  }
  const bbox = toGridBox(region.maskBounds, options.bbox);
  // FLUX.3 reads a box with no coordinate above 1 as 0–1 fractions, so the node refuses [0, 0, 1, 1].
  return bbox && Math.max(...bbox) > 1 ? { bbox, desc } : null;
};

/**
 * FLUX.3 regions: each becomes a row of the box table that ends the caption, boxed around its mask and normalized to
 * the bbox like Ideogram 4's. The base graph's `layout_prompt` appends them at run time, as object rows or, when the
 * builder wired reference images, as edit rows for the output; core metadata then records the caption encoded. Regions
 * the table cannot place are left out, and without any the graph stays untouched.
 */
const addFlux3LayoutRegions = (family: FamilyGraph, options: AddLayoutRegionsOptions): void => {
  const { graph, layoutPrompt } = family;
  if (!layoutPrompt) {
    throw new Error('addFlux3LayoutRegions: base graph is missing the layout prompt node.');
  }
  const regions = options.regions.flatMap((region) => toFlux3Region(region, options) ?? []);
  if (regions.length === 0) {
    return;
  }
  layoutPrompt.regions = regions;
  const metadata = findMetadataNode(graph);
  if (metadata) {
    addEdge(graph, layoutPrompt, 'value', metadata, 'flux3_caption');
  }
};

/** Each layout base places its regions through its own prompt node. */
const LAYOUT_REGION_GRAFTS: Record<
  LayoutRegionalGuidanceBase,
  (family: FamilyGraph, options: AddLayoutRegionsOptions) => void
> = {
  flux3: addFlux3LayoutRegions,
  'ideogram-4': addIdeogram4Regions,
  'moss-image': addMossImageLayoutRegions,
};

/** Throw the first validation reason. */
export const compileCanvasGraph = (input: CompileCanvasGraphInput): CompiledCanvasGraph => {
  const { bbox, compositeImageName, destination, mode, model, projectSettings, strength } = input;
  const processingSize = resolveCanvasProcessingSize(model, input.settings.pidMode, bbox, input.scaling);
  const settings = withProcessingDimensions(input.settings, processingSize);

  const validationReasons = [
    ...getCanvasValidationReasons(input, processingSize),
    ...getGenerationValidationReasons(model, settings, isExternalGenerationMode(mode) ? mode : 'txt2img'),
  ];

  if (validationReasons.length > 0) {
    throw new Error(validationReasons[0]);
  }

  // Canvas outputs are intermediate; gallery outputs are durable.
  const outputIsIntermediate = destination === 'canvas';
  const runtimeProjectSettings = {
    ...projectSettings,
    randDevice: input.randDevice ?? (projectSettings.useCpuNoise ? 'cpu' : 'cuda'),
  };

  if (model.type === 'external_image_generator') {
    // Validation above already named the mode; this narrows the provider inputs to the two modes Canvas compiles.
    if (mode !== 'txt2img' && mode !== 'img2img') {
      throw new Error(`${model.name} does not support ${mode}ing on the canvas.`);
    }

    // The provider is asked for the processing size, one of its own sizes, and the returned whole image is restored
    // to the exact bbox footprint before staging or saving. No local denoising strength applies.
    const backendGraph = buildExternalGraph(
      settings,
      model,
      outputIsIntermediate,
      runtimeProjectSettings,
      mode === 'txt2img' ? { mode } : { initImageName: compositeImageName as string, mode }
    );
    if (mode === 'img2img' && !sizesMatch(processingSize, bbox)) {
      resizeExternalInitImage(backendGraph, compositeImageName as string, processingSize);
    }
    restoreExternalCanvasOutputToBbox(backendGraph, { bbox, destination, mode }, processingSize);
    setCanvasMetadataDimensions(backendGraph, bbox);
    recordCanvasInputs(backendGraph, input);

    return {
      backendGraph,
      graph: toGraphContract(backendGraph, `${model.name} ${mode}`),
      mode,
      negativePromptNodeId: 'negative_prompt',
      positivePromptNodeId: 'positive_prompt',
      seedNodeId: 'seed',
    };
  }

  const builder = getFamilyGraphBuilder(model.base);

  if (!builder) {
    throw new Error(`Canvas generation is not supported for ${model.name}.`);
  }

  const family = builder(settings, model, outputIsIntermediate, runtimeProjectSettings);
  const backendGraph = family.graph;
  const { compositing } = input;

  // Validation above guarantees the composite/mask images required per mode.
  let inpaint: CanvasInpaintSources | undefined;
  if (mode === 'img2img') {
    graftImageToImage(family, model, settings, compositeImageName as string, strength, bbox);
  } else if (mode === 'inpaint') {
    inpaint = graftInpaint(family, input, compositing, processingSize);
  } else if (mode === 'outpaint') {
    inpaint = graftOutpaint(family, input, compositing, processingSize);
  }

  // Every mode accepts already-resolved, separately uploaded control inputs; inpaint and outpaint also hand the
  // adapters what they regenerate.
  if (input.controlLayers && input.controlLayers.length > 0) {
    addControlLayers(backendGraph, {
      base: model.base as SupportedGenerateBase,
      denoise: family.denoise,
      inpaint,
      layers: input.controlLayers,
      modelVariant: model.variant ?? undefined,
    });
  }

  // Regional masks and models must already be validated. Layout bases read only each mask's bounds.
  const regions = input.regionalGuidance ?? [];
  const regionalBase = model.base;
  // The support row answers the layout roles; the base guards narrow the base to the graph tables' keys.
  const regionalSupport = getRegionalGuidanceSupport(regionalBase);
  if (regions.length > 0 && regionalSupport && isRegionalGuidanceSupportedForBase(regionalBase)) {
    if (isLayoutRegionalGuidanceBase(regionalBase)) {
      LAYOUT_REGION_GRAFTS[regionalBase](family, { bbox, layoutKinds: regionalSupport.layoutKinds, regions });
    } else {
      if (!family.conditioning) {
        throw new Error(`Regional guidance has no global conditioning to join in the ${model.name} graph.`);
      }
      addRegionalGuidance(backendGraph, {
        base: regionalBase,
        conditioning: family.conditioning,
        denoise: family.denoise,
        modelVariant: model.variant,
        regions,
        ...(regionalBase === 'krea-2'
          ? {
              transformRegionalPositiveConditioning: (conditioning: BackendInvocationContract, regionId: string) => {
                const seed = backendGraph.nodes.seed;
                if (!seed) {
                  throw new Error('Krea-2 regional guidance requires the base graph seed node.');
                }
                return addKrea2ConditioningEnhancers({
                  conditioning,
                  graph: backendGraph,
                  idPrefix: `rg_krea2_${regionId}`,
                  seed,
                  settings,
                });
              },
            }
          : {}),
      });
    }
  }

  if (mode === 'txt2img' || mode === 'img2img') {
    resizeCanvasOutputToBbox(
      backendGraph,
      bbox,
      processingSize,
      destination,
      shouldUsePidDecode(settings, model.base) && settings.pidMode === 'fit'
    );
  }
  // After every mode has settled its final image node, so the filters see the composited/resized result.
  addOutputPostprocessors(backendGraph, runtimeProjectSettings, outputIsIntermediate);
  setCanvasMetadataDimensions(backendGraph, bbox);
  recordCanvasInputs(backendGraph, input);

  return {
    backendGraph,
    graph: toGraphContract(backendGraph, `${model.name} ${mode}`),
    mode,
    negativePromptNodeId: 'negative_prompt',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};
