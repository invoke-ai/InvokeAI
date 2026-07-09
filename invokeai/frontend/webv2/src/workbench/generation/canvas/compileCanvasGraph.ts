/**
 * Pure canvas generation-graph compiler.
 *
 * {@link compileCanvasGraph} builds the backend graph a canvas invoke submits by
 * grafting image-to-image plumbing onto the existing per-base txt2img builders
 * (`../graph.ts`). It is deliberately pure: no fetch, no engine imports, no
 * React. The executor (Task 16) has already composited + uploaded the bbox
 * source image and passes its name in; this module only shapes nodes/edges.
 *
 * ## Grafting strategy
 *
 * 1. Build the base txt2img graph via `GRAPH_BUILDERS[model.base]` with the
 *    canvas destination (`outputIsIntermediate = true`) and a settings copy whose
 *    width/height are the bbox size (builders read dims from settings).
 * 2. For `img2img`, add a base-appropriate image-to-latents encode node fed by
 *    the composite image + the graph's VAE source, wire its latents into
 *    `denoise_latents.latents`, and set `denoising_start = 1 - strength`.
 * 3. Update `core_metadata` (`generation_mode` → the `img2img` variant, plus
 *    `strength`).
 *
 * Every base in `GRAPH_BUILDERS` has a backend image-to-latents node, so all ten
 * families support img2img (see `CANVAS_I2L_NODE_TYPES`). External image
 * generators have no latent img2img path and are rejected for every canvas mode.
 */

import type { SupportedGenerateBase } from '@workbench/generation/baseGenerationPolicies';
import type { GenerateModelConfig, GenerateSettings } from '@workbench/generation/types';
import type { BackendGraphContract, BackendInvocationContract } from '@workbench/types';
import type { CanvasCompositingSettings, CanvasInfillMethod } from '@workbench/widgets/canvas/invoke/canvasCompositing';

import { getGenerationValidationReasons } from '@workbench/generation/baseGenerationPolicies';
import { GRAPH_BUILDERS } from '@workbench/generation/graph';
import { addEdge, addNode, toGraphContract } from '@workbench/generation/graphBuilder';
import { DEFAULT_CANVAS_COMPOSITING } from '@workbench/widgets/canvas/invoke/canvasCompositing';

import type { CompileCanvasGraphInput, CompiledCanvasGraph } from './types';

import { addControlLayers } from './addControlLayers';
import { addRegionalGuidance, isRegionalGuidanceSupportedForBase } from './addRegionalGuidance';

/**
 * The backend image-to-latents (encode) node type per supported base. sd-1 /
 * sd-2 / sdxl share the SD `i2l` node. Evidence: `invokeai/app/invocations/`
 * and the legacy per-base builders under
 * `features/nodes/util/graph/generation/`.
 */
const CANVAS_I2L_NODE_TYPES: Record<SupportedGenerateBase, string> = {
  'sd-1': 'i2l',
  'sd-2': 'i2l',
  sdxl: 'i2l',
  'sd-3': 'sd3_i2l',
  flux: 'flux_vae_encode',
  flux2: 'flux2_vae_encode',
  cogview4: 'cogview4_i2l',
  'qwen-image': 'qwen_image_i2l',
  'z-image': 'z_image_i2l',
  anima: 'anima_i2l',
};

/**
 * The `denoising_start` for a canvas img2img graft, mirroring legacy
 * `getDenoisingStartAndEnd` (web `graphBuilderUtils.ts`):
 * - sd-3 / flux / flux2 rescale strength with an exponent of 0.2 so the slider's
 *   full (0, 1] range is usable — without it nearly all perceptible change is
 *   crammed into strength > 0.9 (e.g. strength 0.75 → start 0.056, not 0.25).
 * - A FLUX Fill model (`flux` / `dev_fill`) always denoises fully (start = 0).
 * - Every other base stays linear (`start = 1 - strength`).
 *
 * Legacy gates the exponent on an `optimizedDenoisingEnabled` user setting that
 * defaults to `true`; webv2 exposes no such toggle, so the optimized curve is
 * always applied for the eligible bases.
 */
const canvasDenoisingStart = (model: GenerateModelConfig, strength: number): number => {
  if (model.base === 'flux' && model.variant === 'dev_fill') {
    return 0;
  }
  const usesOptimizedCurve = model.base === 'sd-3' || model.base === 'flux' || model.base === 'flux2';
  return 1 - strength ** (usesOptimizedCurve ? 0.2 : 1);
};

/** True for the image-referencing modes (everything but pure txt2img). */
const isImageMode = (mode: CompileCanvasGraphInput['mode']): boolean => mode !== 'txt2img';

/** Canvas-specific validation reasons layered on top of the shared generate ones. */
const getCanvasValidationReasons = (input: CompileCanvasGraphInput): string[] => {
  const { bbox, compositeImageName, maskImageName, mode, model, strength } = input;
  const reasons: string[] = [];

  if (model.type === 'external_image_generator') {
    reasons.push(`${model.name} does not support canvas generation.`);
    return reasons;
  }

  if (!Number.isFinite(bbox.width) || !Number.isFinite(bbox.height) || bbox.width <= 0 || bbox.height <= 0) {
    reasons.push('Canvas bounding box must have a positive area.');
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

/** The settings a base builder sees: identical to the widget's, but sized to the bbox. */
const withBboxDimensions = (settings: GenerateSettings, bbox: CompileCanvasGraphInput['bbox']): GenerateSettings => ({
  ...settings,
  height: bbox.height,
  width: bbox.width,
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

/**
 * Renames a node in place: moves its map key, updates its `id`, and rewires every
 * edge referencing the old id. Used to demote the base `canvas_output` decode to
 * an intermediate `canvas_l2i` so the composite-back node can claim `canvas_output`.
 */
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
  strength: number
): void => {
  const i2lType = requireI2lType(model);
  const denoise = requireDenoise(graph);
  const vaeSource = requireVaeSource(graph);
  const encode = addEncodeNode(graph, i2lType, settings, compositeImageName);

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

/**
 * Shared inpaint/outpaint tail: gradient denoise mask → denoise, expand-with-fade,
 * and the composite-back blend (`canvas_output`). `maskSource` supplies the
 * grayscale mask feeding `create_gradient_mask`; `initialImageName` is the base
 * to paste the decoded result back onto.
 */
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
    /** Either a fixed mask image field or an edge-fed source node. */
    gradientMaskImage?: { image_name: string };
    gradientMaskEdge?: { node: BackendInvocationContract; field: string };
  }
): void => {
  const { compositing, denoise, destination, i2lType, initialImageName, settings, vaeSource } = args;

  // Demote the base decode to an intermediate `canvas_l2i`; the blend becomes `canvas_output`.
  const l2i = renameNode(graph, 'canvas_output', 'canvas_l2i');
  l2i.is_intermediate = true;

  const gradientMask = addNode(graph, {
    coherence_mode: compositing.coherenceMode,
    edge_radius: compositing.coherenceEdgeSize,
    fp32: isSdI2l(i2lType) ? settings.vaePrecision === 'fp32' : false,
    id: 'create_gradient_mask',
    image: { image_name: initialImageName },
    minimum_denoise: compositing.coherenceMinDenoise,
    type: 'create_gradient_mask',
    ...(args.gradientMaskImage ? { mask: args.gradientMaskImage } : {}),
  });

  if (args.gradientMaskEdge) {
    addEdge(graph, args.gradientMaskEdge.node, args.gradientMaskEdge.field, gradientMask, 'mask');
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

  const blend = addNode(graph, {
    id: 'canvas_output',
    is_intermediate: destination === 'canvas',
    layer_base: { image_name: initialImageName },
    type: 'invokeai_img_blend',
    use_cache: false,
  });
  addEdge(graph, l2i, 'image', blend, 'layer_upper');
  addEdge(graph, expandMask, 'image', blend, 'mask');

  // The base builder wired core_metadata → the decode's `metadata`; `renameNode`
  // followed it onto the (now intermediate) canvas_l2i. Re-point it to the final
  // blend so the saved image carries generation metadata (invokeai_img_blend is
  // WithMetadata). Board fields, when present, ride the same node.
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
  imageSourceNode: BackendInvocationContract | null
): { imageField?: { image_name: string }; edgeFrom?: BackendInvocationContract } => {
  if (!noiseMaskImageName) {
    // No noise mask: encode reads the (infilled) initial image directly.
    return imageSourceNode ? { edgeFrom: imageSourceNode } : { imageField: { image_name: imageName } };
  }
  const noise = addNode(graph, {
    amount: 1.0,
    id: 'add_inpaint_noise',
    ...(imageSourceNode ? {} : { image: { image_name: imageName } }),
    mask: { image_name: noiseMaskImageName },
    noise_color: true,
    noise_type: 'gaussian',
    type: 'img_noise',
  });
  addEdge(graph, graph.nodes.seed, 'value', noise, 'seed');
  if (imageSourceNode) {
    addEdge(graph, imageSourceNode, 'image', noise, 'image');
  }
  return { edgeFrom: noise };
};

/** Grafts the inpaint pipeline (content covers the bbox, an inpaint mask restricts it). */
const graftInpaint = (
  graph: BackendGraphContract,
  input: CompileCanvasGraphInput,
  compositing: CanvasCompositingSettings
): void => {
  const { model } = input;
  const i2lType = requireI2lType(model);
  const denoise = requireDenoise(graph);
  const vaeSource = requireVaeSource(graph);
  const initialImageName = input.compositeImageName as string;
  const maskImageName = input.maskImageName as string;
  const strength = input.strength;

  denoise.denoising_start = canvasDenoisingStart(model, strength);
  denoise.denoising_end = 1;

  const encode = addEncodeNode(graph, i2lType, input.settings, null);
  const noiseResult = addNoiseBeforeEncode(graph, initialImageName, input.noiseMaskImageName, null);
  if (noiseResult.edgeFrom) {
    addEdge(graph, noiseResult.edgeFrom, 'image', encode, 'image');
  } else if (noiseResult.imageField) {
    encode.image = noiseResult.imageField;
  }
  addEdge(graph, vaeSource.node, vaeSource.field, encode, 'vae');
  addEdge(graph, encode, 'latents', denoise, 'latents');

  graftMaskTail(graph, {
    compositing,
    denoise,
    destination: input.destination,
    gradientMaskImage: { image_name: maskImageName },
    i2lType,
    initialImageName,
    model,
    settings: input.settings,
    vaeSource,
  });

  setMetadataMode(graph, 'inpaint', strength);
};

/** Grafts the outpaint pipeline (bbox extends past content / has transparent holes). */
const graftOutpaint = (
  graph: BackendGraphContract,
  input: CompileCanvasGraphInput,
  compositing: CanvasCompositingSettings
): void => {
  const { model } = input;
  const i2lType = requireI2lType(model);
  const denoise = requireDenoise(graph);
  const vaeSource = requireVaeSource(graph);
  const initialImageName = input.compositeImageName as string;
  const strength = input.strength;

  denoise.denoising_start = canvasDenoisingStart(model, strength);
  denoise.denoising_end = 1;

  // Infill the transparent region before encode (legacy `getInfill`).
  const infill = addInfillNode(graph, compositing.infillMethod, compositing);
  infill.image = { image_name: initialImageName };

  // Derive a mask from the initial image alpha (transparent → generate), combined
  // with the inpaint mask when one exists.
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

  const encode = addEncodeNode(graph, i2lType, input.settings, null);
  const noiseResult = addNoiseBeforeEncode(graph, initialImageName, input.noiseMaskImageName, infill);
  if (noiseResult.edgeFrom) {
    addEdge(graph, noiseResult.edgeFrom, 'image', encode, 'image');
  }
  addEdge(graph, vaeSource.node, vaeSource.field, encode, 'vae');
  addEdge(graph, encode, 'latents', denoise, 'latents');

  graftMaskTail(graph, {
    compositing,
    denoise,
    destination: input.destination,
    gradientMaskEdge,
    i2lType,
    initialImageName,
    model,
    settings: input.settings,
    vaeSource,
  });

  setMetadataMode(graph, 'outpaint', strength);
};

/**
 * Compiles a canvas invoke into a backend graph. Throws a validation `Error`
 * (message = first offending reason) for unsupported models/modes or bad
 * geometry, mirroring `compileGenerateGraph`.
 */
export const compileCanvasGraph = (input: CompileCanvasGraphInput): CompiledCanvasGraph => {
  const { bbox, compositeImageName, destination, mode, model, projectSettings, strength } = input;
  const settings = withBboxDimensions(input.settings, bbox);

  const validationReasons = [...getCanvasValidationReasons(input), ...getGenerationValidationReasons(model, settings)];

  if (validationReasons.length > 0) {
    throw new Error(validationReasons[0]);
  }

  const builder = GRAPH_BUILDERS[model.base as SupportedGenerateBase];

  if (!builder || model.type === 'external_image_generator') {
    throw new Error(`${model.name} does not support canvas generation.`);
  }

  // Mirror compileGenerateGraph: only a `canvas` destination stages an
  // intermediate output; `gallery` produces a durable image.
  const outputIsIntermediate = destination === 'canvas';
  const backendGraph = builder(settings, model, outputIsIntermediate, projectSettings);
  const compositing = input.compositing ?? DEFAULT_CANVAS_COMPOSITING;

  // Validation above guarantees the composite/mask images required per mode.
  if (mode === 'img2img') {
    graftImageToImage(backendGraph, model, settings, compositeImageName as string, strength);
  } else if (mode === 'inpaint') {
    graftInpaint(backendGraph, input, compositing);
  } else if (mode === 'outpaint') {
    graftOutpaint(backendGraph, input, compositing);
  }

  // Control layers apply in every mode (legacy allows control with all). The
  // executor already composited + uploaded each layer separately and resolved
  // its model; the caller passes only valid layers.
  if (input.controlLayers && input.controlLayers.length > 0) {
    addControlLayers(backendGraph, {
      base: model.base as SupportedGenerateBase,
      layers: input.controlLayers,
      modelVariant: model.variant ?? undefined,
    });
  }

  // Regional guidance applies in every mode too. The executor already composited
  // + uploaded each region's mask and resolved its reference-image models; it
  // passes only valid regions for a supported base (sd-1 / sdxl / flux).
  if (input.regionalGuidance && input.regionalGuidance.length > 0 && isRegionalGuidanceSupportedForBase(model.base)) {
    addRegionalGuidance(backendGraph, { base: model.base, regions: input.regionalGuidance });
  }

  return {
    backendGraph,
    graph: toGraphContract(backendGraph, `${model.name} ${mode}`),
    mode,
    negativePromptNodeId: 'negative_prompt',
    positivePromptNodeId: 'positive_prompt',
    seedNodeId: 'seed',
  };
};
