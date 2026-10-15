/**
 * Pure canvas compiler over uploaded inputs. Resize through the processing grid and restore the bbox; reject
 * external models and unsupported encode modes.
 */

import type { SupportedGenerateBase } from '@features/generation/core/baseGenerationPolicies';
import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { getGenerationValidationReasons } from '@features/generation/core/baseGenerationPolicies';
import { GRAPH_BUILDERS } from '@features/generation/core/graph';
import { addEdge, addNode, toGraphContract } from '@features/generation/core/graphBuilder';
import { addKrea2ConditioningEnhancers } from '@features/generation/core/krea2Conditioning';
import { getIsPidSupportedBase } from '@features/generation/core/pid';

import type {
  CanvasCompositingSettings,
  CanvasInfillMethod,
  CompileCanvasGraphInput,
  CompiledCanvasGraph,
} from './types';

import { addControlLayers } from './addControlLayers';
import { addRegionalGuidance, isRegionalGuidanceSupportedForBase } from './addRegionalGuidance';
import { type CanvasSize, resolveCanvasProcessingSize } from './canvasProcessingSize';

/** Map bases to encode nodes; SD families share i2l. */
const CANVAS_I2L_NODE_TYPES: Partial<Record<SupportedGenerateBase, string>> = {
  'sd-1': 'i2l',
  'sd-2': 'i2l',
  sdxl: 'i2l',
  'sd-3': 'sd3_i2l',
  flux: 'flux_vae_encode',
  flux2: 'flux2_vae_encode',
  cogview4: 'cogview4_i2l',
  'qwen-image': 'qwen_image_i2l',
  'z-image': 'z_image_i2l',
  // Krea-2 shares the Qwen-Image VAE, so it encodes with that family's node too.
  'krea-2': 'qwen_image_i2l',
  anima: 'anima_i2l',
  wan: 'wan_i2l',
  // Ideogram lacks an encode node; reject these modes with an actionable error.
};

/** SD-3, FLUX, and FLUX.2 use start = 1 - strength^0.2. FLUX Fill always starts at 0; other bases use 1 - strength. */
const canvasDenoisingStart = (model: GenerateModelConfig, strength: number): number => {
  if (model.base === 'flux' && model.variant === 'dev_fill') {
    return 0;
  }
  const usesOptimizedCurve = model.base === 'sd-3' || model.base === 'flux' || model.base === 'flux2';
  return 1 - strength ** (usesOptimizedCurve ? 0.2 : 1);
};

/** True for the image-referencing modes (everything but pure txt2img). */
const isImageMode = (mode: CompileCanvasGraphInput['mode']): boolean => mode !== 'txt2img';

const sizesMatch = (left: CanvasSize, right: CanvasSize): boolean =>
  left.width === right.width && left.height === right.height;

/** Canvas-specific validation reasons layered on top of the shared generate ones. */
const getCanvasValidationReasons = (input: CompileCanvasGraphInput): string[] => {
  const { bbox, compositeImageName, maskImageName, mode, model, strength } = input;
  const reasons: string[] = [];

  if (model.type === 'external_image_generator') {
    reasons.push(`${model.name} does not support canvas generation.`);
    return reasons;
  }

  // Reject canvas PiD because its graph lacks the expected VAE/decode seam.
  if (input.settings.pidMode !== 'off' && getIsPidSupportedBase(model.base)) {
    reasons.push('PiD decoding is not supported on the canvas yet. Turn PiD off to generate here.');
  }

  if (!Number.isFinite(bbox.width) || !Number.isFinite(bbox.height) || bbox.width <= 0 || bbox.height <= 0) {
    reasons.push('Canvas bounding box must have a positive area.');
  } else if (!Number.isInteger(bbox.width) || !Number.isInteger(bbox.height)) {
    reasons.push('Canvas bounding box dimensions must be whole pixels.');
  }

  if (isImageMode(mode)) {
    if (!compositeImageName) {
      reasons.push('Canvas generation requires a composited source image.');
    }

    if (!Number.isFinite(strength) || strength <= 0 || strength > 1) {
      reasons.push('Canvas denoising strength must be greater than 0 and at most 1.');
    }
  }

  // Inpaint always needs a mask (an active inpaint mask defines the region);
  // outpaint derives its mask from the raster alpha, so its mask is optional.
  if (mode === 'inpaint' && !maskImageName) {
    reasons.push('Canvas inpainting requires an inpaint mask.');
  }

  return reasons;
};

/** The settings a base builder sees: identical to the widget's, but sized to the model-valid processing frame. */
const withProcessingDimensions = (settings: GenerateSettings, processingSize: CanvasSize): GenerateSettings => ({
  ...settings,
  height: processingSize.height,
  width: processingSize.width,
});

/** Locates the node + field feeding a decode/denoise input edge (e.g. `canvas_output.vae`). */
const findEdgeSource = (
  graph: BackendGraphContract,
  destNodeId: string,
  destField: string
): { node: BackendInvocationContract; field: string } | null => {
  const edge = graph.edges.find(
    (candidate) => candidate.destination.node_id === destNodeId && candidate.destination.field === destField
  );
  const node = edge ? graph.nodes[edge.source.node_id] : undefined;
  return edge && node ? { field: edge.source.field, node } : null;
};

/** Resolves the VAE source feeding the graph's decode node (throws if absent). */
const requireVaeSource = (graph: BackendGraphContract): { node: BackendInvocationContract; field: string } => {
  const source = findEdgeSource(graph, 'canvas_output', 'vae');
  if (!source) {
    throw new Error('Canvas generation could not resolve a VAE source in the base graph.');
  }
  return source;
};

/** Rename every matching node ID and edge reference. */
const renameNode = (graph: BackendGraphContract, oldId: string, newId: string): BackendInvocationContract => {
  const node = graph.nodes[oldId];
  if (!node) {
    throw new Error(`Canvas generation could not find the "${oldId}" node to rename.`);
  }
  delete graph.nodes[oldId];
  node.id = newId;
  graph.nodes[newId] = node;
  for (const edge of graph.edges) {
    if (edge.source.node_id === oldId) {
      edge.source.node_id = newId;
    }
    if (edge.destination.node_id === oldId) {
      edge.destination.node_id = newId;
    }
  }
  return node;
};

/** Resizes a model-grid processing output back to the bbox's exact final footprint. */
const resizeCanvasOutputToBbox = (
  graph: BackendGraphContract,
  bbox: CompileCanvasGraphInput['bbox'],
  processingSize: CanvasSize,
  destination: CompileCanvasGraphInput['destination']
): void => {
  if (sizesMatch(bbox, processingSize)) {
    return;
  }

  const processingOutput = renameNode(graph, 'canvas_output', 'canvas_processing_output');
  processingOutput.is_intermediate = true;

  const output = addNode(graph, {
    height: bbox.height,
    id: 'canvas_output',
    is_intermediate: destination === 'canvas',
    type: 'img_resize',
    use_cache: false,
    width: bbox.width,
  });
  addEdge(graph, processingOutput, 'image', output, 'image');

  const metadataEdge = graph.edges.find(
    (edge) => edge.destination.node_id === processingOutput.id && edge.destination.field === 'metadata'
  );
  if (metadataEdge) {
    metadataEdge.destination.node_id = output.id;
  }
};

/** Keeps saved-image metadata aligned to the bbox footprint, not the internal processing size. */
const setCanvasMetadataDimensions = (graph: BackendGraphContract, bbox: CompileCanvasGraphInput['bbox']): void => {
  const metadata = Object.values(graph.nodes).find((node) => node.type === 'core_metadata');
  if (metadata) {
    metadata.width = bbox.width;
    metadata.height = bbox.height;
  }
};

/** Sets the metadata `generation_mode` variant + strength (legacy parity). */
const setMetadataMode = (graph: BackendGraphContract, mode: string, strength: number): void => {
  const metadata = Object.values(graph.nodes).find((node) => node.type === 'core_metadata');
  if (metadata) {
    if (typeof metadata.generation_mode === 'string') {
      metadata.generation_mode = metadata.generation_mode.replace('txt2img', mode);
    }
    metadata.strength = strength;
  }
};

/** The SD `i2l` node carries fp32; every other family's encode node does not. */
const isSdI2l = (i2lType: string): boolean => i2lType === 'i2l';

/** Adds the base-appropriate image-to-latents encode node fed by `imageName`. */
const addEncodeNode = (
  graph: BackendGraphContract,
  i2lType: string,
  settings: GenerateSettings,
  imageName: string | null
): BackendInvocationContract =>
  addNode(graph, {
    id: 'canvas_i2l',
    ...(imageName ? { image: { image_name: imageName } } : {}),
    type: i2lType,
    ...(isSdI2l(i2lType) ? { fp32: settings.vaePrecision === 'fp32' } : {}),
  });

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

/** Resolves the encode node type for a supported base (throws otherwise). */
const requireI2lType = (model: GenerateModelConfig): string => {
  if (model.type === 'external_image_generator') {
    throw new Error(`${model.name} does not support canvas generation.`);
  }
  const i2lType = CANVAS_I2L_NODE_TYPES[model.base as SupportedGenerateBase];
  if (!i2lType) {
    throw new Error(`Canvas generation is not supported for ${model.name}.`);
  }
  return i2lType;
};

/** Resolves the base graph's denoise node (throws if absent). */
const requireDenoise = (graph: BackendGraphContract): BackendInvocationContract => {
  const denoise = graph.nodes.denoise_latents;
  if (!denoise) {
    throw new Error('Canvas generation could not find the denoise node in the base graph.');
  }
  return denoise;
};

/** Grafts the image-to-latents encode path onto a freshly built base graph (img2img). */
const graftImageToImage = (
  graph: BackendGraphContract,
  model: GenerateModelConfig,
  settings: GenerateSettings,
  compositeImageName: string,
  strength: number,
  bbox: CompileCanvasGraphInput['bbox'],
  processingSize: CanvasSize
): void => {
  const i2lType = requireI2lType(model);
  const denoise = requireDenoise(graph);
  const vaeSource = requireVaeSource(graph);
  const needsResize = !sizesMatch(bbox, processingSize);
  const encode = addEncodeNode(graph, i2lType, settings, needsResize ? null : compositeImageName);

  if (needsResize) {
    const resize = addInputResizeNode(graph, 'canvas_resize_initial_to_processing', compositeImageName, processingSize);
    addEdge(graph, resize, 'image', encode, 'image');
  }

  addEdge(graph, vaeSource.node, vaeSource.field, encode, 'vae');
  addEdge(graph, encode, 'latents', denoise, 'latents');
  denoise.denoising_start = canvasDenoisingStart(model, strength);
  denoise.denoising_end = 1;

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

/** outputOnlyMaskedRegions selects transparent output versus compositing with the original. */
const graftMaskTail = (
  graph: BackendGraphContract,
  args: {
    model: GenerateModelConfig;
    settings: GenerateSettings;
    i2lType: string;
    denoise: BackendInvocationContract;
    vaeSource: { node: BackendInvocationContract; field: string };
    initialImageName: string;
    compositing: CanvasCompositingSettings;
    destination: CompileCanvasGraphInput['destination'];
    bbox: CompileCanvasGraphInput['bbox'];
    processingSize: CanvasSize;
    gradientImageEdge?: { node: BackendInvocationContract; field: string };
    /** Either a fixed mask image field or an edge-fed source node. */
    gradientMaskImage?: { image_name: string };
    gradientMaskEdge?: { node: BackendInvocationContract; field: string };
  }
): void => {
  const { compositing, denoise, destination, i2lType, initialImageName, settings, vaeSource } = args;
  const needsResize = !sizesMatch(args.bbox, args.processingSize);

  // The final output owns canvas_output; the base decoder becomes intermediate.
  const l2i = renameNode(graph, 'canvas_output', 'canvas_l2i');
  l2i.is_intermediate = true;

  const gradientMask = addNode(graph, {
    coherence_mode: compositing.coherenceMode,
    edge_radius: compositing.coherenceEdgeSize,
    fp32: isSdI2l(i2lType) ? settings.vaePrecision === 'fp32' : false,
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
  addEdge(graph, vaeSource.node, vaeSource.field, gradientMask, 'vae');
  // The optional UNet edge only applies to SD-family models (legacy `isMainModelWithoutUnet`).
  const unetSource = findEdgeSource(graph, 'denoise_latents', 'unet');
  if (unetSource) {
    addEdge(graph, unetSource.node, unetSource.field, gradientMask, 'unet');
  }
  addEdge(graph, gradientMask, 'denoise_mask', denoise, 'denoise_mask');

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

  const output = compositing.outputOnlyMaskedRegions
    ? addNode(graph, {
        id: 'canvas_output',
        invert_mask: true,
        is_intermediate: destination === 'canvas',
        type: 'apply_mask_to_image',
        use_cache: false,
      })
    : addNode(graph, {
        id: 'canvas_output',
        is_intermediate: destination === 'canvas',
        layer_base: { image_name: initialImageName },
        type: 'invokeai_img_blend',
        use_cache: false,
      });
  addEdge(graph, generatedImageSource, 'image', output, compositing.outputOnlyMaskedRegions ? 'image' : 'layer_upper');
  addEdge(graph, outputMaskSource, 'image', output, 'mask');

  // Move metadata to the final saved output after renaming.
  const metadataEdge = graph.edges.find(
    (edge) => edge.destination.node_id === 'canvas_l2i' && edge.destination.field === 'metadata'
  );
  if (metadataEdge) {
    metadataEdge.destination.node_id = 'canvas_output';
  }
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
  graph: BackendGraphContract,
  input: CompileCanvasGraphInput,
  compositing: CanvasCompositingSettings,
  processingSize: CanvasSize
): void => {
  const { model } = input;
  const i2lType = requireI2lType(model);
  const denoise = requireDenoise(graph);
  const vaeSource = requireVaeSource(graph);
  const initialImageName = input.compositeImageName as string;
  const maskImageName = input.maskImageName as string;
  const strength = input.strength;
  const needsResize = !sizesMatch(input.bbox, processingSize);

  denoise.denoising_start = canvasDenoisingStart(model, strength);
  denoise.denoising_end = 1;

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

  const encode = addEncodeNode(graph, i2lType, input.settings, null);
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
  addEdge(graph, vaeSource.node, vaeSource.field, encode, 'vae');
  addEdge(graph, encode, 'latents', denoise, 'latents');

  graftMaskTail(graph, {
    bbox: input.bbox,
    compositing,
    denoise,
    destination: input.destination,
    ...(initialResize ? { gradientImageEdge: { field: 'image', node: initialResize } } : {}),
    ...(maskResize
      ? { gradientMaskEdge: { field: 'image', node: maskResize } }
      : { gradientMaskImage: { image_name: maskImageName } }),
    i2lType,
    initialImageName,
    model,
    processingSize,
    settings: input.settings,
    vaeSource,
  });

  setMetadataMode(graph, 'inpaint', strength);
};

/** Grafts the outpaint pipeline (bbox extends past content / has transparent holes). */
const graftOutpaint = (
  graph: BackendGraphContract,
  input: CompileCanvasGraphInput,
  compositing: CanvasCompositingSettings,
  processingSize: CanvasSize
): void => {
  const { model } = input;
  const i2lType = requireI2lType(model);
  const denoise = requireDenoise(graph);
  const vaeSource = requireVaeSource(graph);
  const initialImageName = input.compositeImageName as string;
  const strength = input.strength;
  const needsResize = !sizesMatch(input.bbox, processingSize);

  denoise.denoising_start = canvasDenoisingStart(model, strength);
  denoise.denoising_end = 1;

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

  const encode = addEncodeNode(graph, i2lType, input.settings, null);
  const noiseResult = addNoiseBeforeEncode(graph, initialImageName, input.noiseMaskImageName, infill, noiseMaskResize);
  if (noiseResult.edgeFrom) {
    addEdge(graph, noiseResult.edgeFrom, 'image', encode, 'image');
  }
  addEdge(graph, vaeSource.node, vaeSource.field, encode, 'vae');
  addEdge(graph, encode, 'latents', denoise, 'latents');

  graftMaskTail(graph, {
    bbox: input.bbox,
    compositing,
    denoise,
    destination: input.destination,
    ...(needsResize ? { gradientImageEdge: { field: 'image', node: infill } } : {}),
    gradientMaskEdge,
    i2lType,
    initialImageName,
    model,
    processingSize,
    settings: input.settings,
    vaeSource,
  });

  setMetadataMode(graph, 'outpaint', strength);
};

/** Throw the first validation reason. */
export const compileCanvasGraph = (input: CompileCanvasGraphInput): CompiledCanvasGraph => {
  const { bbox, compositeImageName, destination, mode, model, projectSettings, strength } = input;
  const processingSize = resolveCanvasProcessingSize(model, input.settings.pidMode, bbox, input.scaling);
  const settings = withProcessingDimensions(input.settings, processingSize);

  const validationReasons = [...getCanvasValidationReasons(input), ...getGenerationValidationReasons(model, settings)];

  if (validationReasons.length > 0) {
    throw new Error(validationReasons[0]);
  }

  const builder = GRAPH_BUILDERS[model.base as SupportedGenerateBase];

  if (!builder || model.type === 'external_image_generator') {
    throw new Error(`${model.name} does not support canvas generation.`);
  }

  // Canvas outputs are intermediate; gallery outputs are durable.
  const outputIsIntermediate = destination === 'canvas';
  const runtimeProjectSettings = {
    ...projectSettings,
    randDevice: input.randDevice ?? (projectSettings.useCpuNoise ? 'cpu' : 'cuda'),
  };
  const backendGraph = builder(settings, model, outputIsIntermediate, runtimeProjectSettings);
  const { compositing } = input;

  // Validation above guarantees the composite/mask images required per mode.
  if (mode === 'img2img') {
    graftImageToImage(backendGraph, model, settings, compositeImageName as string, strength, bbox, processingSize);
  } else if (mode === 'inpaint') {
    graftInpaint(backendGraph, input, compositing, processingSize);
  } else if (mode === 'outpaint') {
    graftOutpaint(backendGraph, input, compositing, processingSize);
  }

  // Every mode accepts already-resolved, separately uploaded control inputs.
  if (input.controlLayers && input.controlLayers.length > 0) {
    addControlLayers(backendGraph, {
      base: model.base as SupportedGenerateBase,
      layers: input.controlLayers,
      modelVariant: model.variant ?? undefined,
    });
  }

  // Regional masks and models must already be validated.
  if (input.regionalGuidance && input.regionalGuidance.length > 0 && isRegionalGuidanceSupportedForBase(model.base)) {
    addRegionalGuidance(backendGraph, {
      base: model.base,
      modelVariant: model.variant,
      regions: input.regionalGuidance,
      ...(model.base === 'krea-2'
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

  if (mode === 'txt2img' || mode === 'img2img') {
    resizeCanvasOutputToBbox(backendGraph, bbox, processingSize, destination);
  }
  setCanvasMetadataDimensions(backendGraph, bbox);

  return {
    backendGraph,
    graph: toGraphContract(backendGraph, `${model.name} ${mode}`),
    mode,
    negativePromptNodeId: 'negative_prompt',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};
