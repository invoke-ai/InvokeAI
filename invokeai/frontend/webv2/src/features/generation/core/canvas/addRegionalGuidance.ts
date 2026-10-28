import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';

import { getArchitectureFeatures } from '@features/generation/core/architectureCapabilities';
import { addEdge, addNode } from '@features/generation/core/graphBuilder';
import { COMPEL_PROMPT_NODE_TYPES } from '@features/generation/core/prompt/compel';
import { unwrapPromptFence } from '@features/generation/core/prompt/jsonPrompt';

import type { Rect } from './types';

/** The deterministic denoise node id every canvas base graph uses. */
const DENOISE_NODE_ID = 'denoise_latents';
/** The deterministic global positive/negative conditioning + collector node ids. */
const POS_COND_ID = 'pos_cond';
const NEG_COND_ID = 'neg_cond';
const POS_COND_COLLECT_ID = 'pos_cond_collect';
const NEG_COND_COLLECT_ID = 'neg_cond_collect';

/** The bases whose text encoder accepts a region mask and whose denoiser accepts a conditioning list. */
export type MaskRegionalGuidanceBase = 'sd-1' | 'sd-2' | 'sdxl' | 'flux' | 'flux2' | 'krea-2' | 'z-image' | 'anima';
/** The bases that read regions as boxes of a structured layout prompt; a mask contributes only its bounds. */
export type LayoutRegionalGuidanceBase = 'moss-image';
export type RegionalGuidanceBase = MaskRegionalGuidanceBase | LayoutRegionalGuidanceBase;

/** How a base consumes a region: as masked conditioning, or as a layout box around the mask. */
export type RegionalGuidanceMode = 'masks' | 'layout';

/** The reference-image node kind a base's regions can consume. */
export type RegionalReferenceImageKind = 'ip_adapter' | 'flux_redux';

/** What the backend honours per region for one base, beyond the masked positive prompt. */
export interface RegionalGuidanceSupport {
  /** Display name used in rejection reasons and settings hints. */
  label: string;
  mode: RegionalGuidanceMode;
  /** Masked regional negative prompts reach the negative collector. */
  negativePrompt: boolean;
  /** The positive prompt can be re-encoded over the inverted mask as a negative. */
  autoNegative: boolean;
  /** Mask-scoped reference images, or `null` when the denoiser has no regional image path. */
  referenceImages: RegionalReferenceImageKind | null;
}

type MaskSupportRow = Pick<RegionalGuidanceSupport, 'label' | 'referenceImages'> & { mode: 'masks' };
type LayoutSupportRow = { label: string; mode: 'layout'; referenceImages: null };

/** Graph code owns node mapping and each base's mode; the backend owns regional-negative support. */
const REGIONAL_GUIDANCE_SUPPORT: Record<MaskRegionalGuidanceBase, MaskSupportRow> &
  Record<LayoutRegionalGuidanceBase, LayoutSupportRow> = {
  'sd-1': { label: 'SD 1.x', mode: 'masks', referenceImages: 'ip_adapter' },
  'sd-2': { label: 'SD 2.x', mode: 'masks', referenceImages: 'ip_adapter' },
  sdxl: { label: 'SDXL', mode: 'masks', referenceImages: 'ip_adapter' },
  flux: { label: 'FLUX', mode: 'masks', referenceImages: 'flux_redux' },
  flux2: { label: 'FLUX.2', mode: 'masks', referenceImages: null },
  'krea-2': { label: 'Krea-2', mode: 'masks', referenceImages: null },
  'z-image': { label: 'Z-Image', mode: 'masks', referenceImages: null },
  anima: { label: 'Anima', mode: 'masks', referenceImages: null },
  'moss-image': { label: 'MOSS-Image', mode: 'layout', referenceImages: null },
};

/** Frontend graph keys are narrower than backend capability answers. */
export const isRegionalGuidanceSupportedForBase = (base: string): base is RegionalGuidanceBase =>
  // Require both backend support and a local graph row before narrowing the type.
  (getArchitectureFeatures(base)?.supports_regional_guidance ?? false) && base in REGIONAL_GUIDANCE_SUPPORT;

/** Whether a supported base places regions as layout boxes rather than masking their conditioning. */
export const isLayoutRegionalGuidanceBase = (base: RegionalGuidanceBase): base is LayoutRegionalGuidanceBase =>
  REGIONAL_GUIDANCE_SUPPORT[base].mode === 'layout';

/** Auto-negative requires negative-mask support; a layout prompt has no negative side. */
export const getRegionalGuidanceSupport = (base: string | null): RegionalGuidanceSupport | null => {
  if (base === null || !isRegionalGuidanceSupportedForBase(base)) {
    return null;
  }

  const row = REGIONAL_GUIDANCE_SUPPORT[base];
  const negative = row.mode === 'masks' && (getArchitectureFeatures(base)?.regional_negative ?? false);

  return { ...row, autoNegative: negative, negativePrompt: negative };
};

/** A region's role in a layout prompt; the canvas persists a subset of these and Canvas checks it does. */
export type RegionalLayoutKind = 'object' | 'character' | 'text';

export interface RegionalLayout {
  kind: RegionalLayoutKind;
  /** The exact text a `text` region renders; the other kinds ignore it. */
  text?: string;
}

/** What keeps a layout base from placing a region. */
export type LayoutRegionIssue = 'missing_prompt' | 'missing_text';

/**
 * Why a layout base leaves a region out, or null when it can place it: every layout element needs a caption, and a
 * text element its visible text. Whitespace alone counts as missing.
 */
export const getLayoutRegionIssue = (region: {
  positivePrompt: string | null;
  layout?: RegionalLayout;
}): LayoutRegionIssue | null => {
  if (!region.positivePrompt?.trim()) {
    return 'missing_prompt';
  }
  if (region.layout?.kind === 'text' && !region.layout.text?.trim()) {
    return 'missing_text';
  }
  return null;
};

/**
 * Why a layout base cannot lay these regions out, or null: the layout is built around the global prompt (as submitted,
 * template applied), so regions it can place need one. Readiness asks before anything uploads; the canvas compiler
 * asks again for the regions it actually places.
 */
export const getLayoutPromptRejectionReason = (params: {
  base: string;
  prompt: string;
  regions: readonly { positivePrompt: string | null; layout?: RegionalLayout }[];
}): string | null => {
  const support = getRegionalGuidanceSupport(params.base);
  if (
    support?.mode !== 'layout' ||
    params.prompt.trim().length > 0 ||
    !params.regions.some((region) => getLayoutRegionIssue(region) === null)
  ) {
    return null;
  }
  return `${support.label} needs a prompt to place regional guidance in its layout. Enter a prompt or disable the regional guidance layers.`;
};

const LOOKS_LIKE_LAYOUT_DRAFT = /"bbox_coord_space"\s*:\s*"abs"/;
const LAYOUT_DRAFT_FIX = 'Run Expand Prompt again or fix it.';

/**
 * Why a MOSS-Image prompt holding a layout draft (as Expand Prompt writes it: a JSON object with
 * `bbox_coord_space: "abs"`, optionally in a ``` fence) cannot be compiled, or null. Plain text and compiled layouts
 * (`abs_centralized`) pass. Only the shape the layout node needs to start is checked; the backend validates the boxes.
 */
export const getLayoutDraftRejectionReason = (params: { base: string; prompt: string }): string | null => {
  if (getRegionalGuidanceSupport(params.base)?.mode !== 'layout') {
    return null;
  }
  const body = unwrapPromptFence(params.prompt);
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    return body.trimStart().startsWith('{') && LOOKS_LIKE_LAYOUT_DRAFT.test(body)
      ? `The prompt looks like a MOSS-Image layout draft but is not valid JSON. ${LAYOUT_DRAFT_FIX}`
      : null;
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return null;
  }
  const draft = parsed as Record<string, unknown>;
  if (draft.bbox_coord_space !== 'abs') {
    return null;
  }
  if (draft.bbox_format !== 'xyxy') {
    return `The MOSS-Image layout draft in the prompt uses a bbox_format other than xyxy. ${LAYOUT_DRAFT_FIX}`;
  }
  const summary =
    typeof draft.summary === 'object' && draft.summary !== null ? (draft.summary as Record<string, unknown>) : null;
  if (
    typeof summary?.caption !== 'string' ||
    !summary.caption.trim() ||
    !Array.isArray(draft.items) ||
    draft.items.length === 0 ||
    typeof draft.width !== 'number' ||
    typeof draft.height !== 'number'
  ) {
    return `The MOSS-Image layout draft in the prompt needs a summary with a caption, at least one item, and a numeric width and height. ${LAYOUT_DRAFT_FIX}`;
  }
  return null;
};

/** A resolved reference-image (component) model identifier — the backend model field shape. */
export interface RegionalReferenceModel {
  key: string;
  hash?: string;
  name: string;
  base: string;
  type: string;
}

/** A regional `ip_adapter` reference image (SD1 / SDXL). */
export interface RegionalIPAdapterInput {
  type: 'ip_adapter';
  /** The reference-image id (mints the deterministic `ip_adapter_${id}` node id). */
  id: string;
  imageName: string;
  model: RegionalReferenceModel;
  weight: number;
  method: string;
  clipVisionModel: string;
  beginEndStepPct: [number, number];
}

/** A regional `flux_redux` reference image (FLUX). */
export interface RegionalFluxReduxInput {
  type: 'flux_redux';
  /** The reference-image id (mints the deterministic `flux_redux_${id}` node id). */
  id: string;
  imageName: string;
  model: RegionalReferenceModel;
  /** Backend redux knobs (downsampling_factor + weight), resolved from imageInfluence upstream. */
  settings: { downsampling_factor: number; weight: number };
}

/** One resolved regional reference image. */
export type RegionalReferenceImageInput = RegionalIPAdapterInput | RegionalFluxReduxInput;

/** One fully-resolved region's graph contribution (invalid regions filtered out first). */
export interface RegionalGuidanceInput {
  /** The document layer id (mints deterministic node ids). */
  id: string;
  /** The uploaded per-region mask image name (alpha = region coverage). */
  maskImageName: string;
  positivePrompt: string | null;
  negativePrompt: string | null;
  autoNegative: boolean;
  referenceImages: readonly RegionalReferenceImageInput[];
  /** The mask's alpha bounds in bbox pixels, origin at the bbox corner; layout bases place the region by them. */
  maskBounds: Rect;
  /** Absent means an object region; only layout bases read it. */
  layout?: RegionalLayout;
}

/** Options for {@link addRegionalGuidance}. */
export interface AddRegionalGuidanceOptions {
  base: MaskRegionalGuidanceBase;
  modelVariant?: string | null;
  regions: readonly RegionalGuidanceInput[];
  /** Applies model-specific transforms before a regional positive conditioning is collected. */
  transformRegionalPositiveConditioning?: (
    conditioning: BackendInvocationContract,
    regionId: string
  ) => BackendInvocationContract;
}

/** Per-base conditioning encoder node type + the fields carrying the prompt. */
const conditioningNodeType = (base: MaskRegionalGuidanceBase, modelVariant?: string | null): string => {
  switch (base) {
    case 'sd-1':
    case 'sd-2':
    case 'sdxl':
      return COMPEL_PROMPT_NODE_TYPES[base];
    case 'flux':
      return 'flux_text_encoder';
    case 'flux2':
      return modelVariant === 'dev' ? 'flux2_dev_text_encoder' : 'flux2_klein_text_encoder';
    case 'krea-2':
      return 'krea2_text_encoder';
    case 'z-image':
      return 'z_image_text_encoder';
    case 'anima':
      return 'anima_text_encoder';
  }
};

/** The prompt input fields to set on a regional conditioning node (SDXL mirrors prompt→style). */
const promptFields = (base: MaskRegionalGuidanceBase): readonly string[] =>
  base === 'sdxl' ? ['prompt', 'style'] : ['prompt'];

/** Copy global encoder fields; wire masks separately. */
const copyEncoderFields = (base: MaskRegionalGuidanceBase, modelVariant?: string | null): readonly string[] => {
  switch (base) {
    case 'sdxl':
      return ['clip', 'clip2'];
    case 'flux':
      return ['clip', 't5_encoder', 't5_max_seq_len'];
    case 'flux2':
      return modelVariant === 'dev' ? ['mistral_encoder', 'max_seq_len'] : ['qwen3_encoder', 'max_seq_len'];
    case 'krea-2':
      return ['qwen3_vl_encoder'];
    case 'z-image':
      return ['qwen3_encoder'];
    case 'anima':
      // Copied only where the global prompt has it: Anima-3.8B's second encoder.
      return ['qwen3_encoder', 'qwen3_5_encoder'];
    case 'sd-1':
    case 'sd-2':
      return ['clip'];
  }
};

const copyEncoderEdges = (
  graph: BackendGraphContract,
  sourceNodeId: string,
  target: BackendInvocationContract,
  fields: readonly string[]
): void => {
  const fieldSet = new Set(fields);
  for (const edge of graph.edges) {
    if (edge.destination.node_id !== sourceNodeId || !fieldSet.has(edge.destination.field)) {
      continue;
    }
    graph.edges.push({
      destination: { field: edge.destination.field, node_id: target.id },
      source: { field: edge.source.field, node_id: edge.source.node_id },
    });
  }
};

/** Resolves (or lazily creates) the collector feeding `denoise.<field>`, with a stable fallback id. */
const resolveDenoiseCollector = (
  graph: BackendGraphContract,
  denoise: BackendInvocationContract,
  denoiseField: string,
  fallbackId: string
): BackendInvocationContract => {
  const existing = graph.edges.find(
    (edge) => edge.destination.node_id === denoise.id && edge.destination.field === denoiseField
  );
  if (existing) {
    const node = graph.nodes[existing.source.node_id];
    if (node) {
      return node;
    }
  }
  const collector = addNode(graph, { id: fallbackId, type: 'collect' });
  addEdge(graph, collector, 'collection', denoise, denoiseField);
  return collector;
};

/** Builds a per-region conditioning node with its prompt set and encoder edges copied. */
const addRegionalConditioning = (
  graph: BackendGraphContract,
  base: MaskRegionalGuidanceBase,
  modelVariant: string | null | undefined,
  nodeId: string,
  prompt: string,
  copyFrom: string
): BackendInvocationContract => {
  const node = addNode(graph, { id: nodeId, type: conditioningNodeType(base, modelVariant) });
  for (const field of promptFields(base)) {
    (node as Record<string, unknown>)[field] = prompt;
  }
  copyEncoderEdges(graph, copyFrom, node, copyEncoderFields(base, modelVariant));
  return node;
};

/** Masked conditioning for mask bases. Inputs must be prevalidated and gated by regional support. */
export const addRegionalGuidance = (graph: BackendGraphContract, options: AddRegionalGuidanceOptions): void => {
  const { base, modelVariant, regions, transformRegionalPositiveConditioning } = options;
  const denoise = graph.nodes[DENOISE_NODE_ID];
  if (!denoise) {
    throw new Error('addRegionalGuidance: base graph is missing the denoise node.');
  }
  const posCondCollect = graph.nodes[POS_COND_COLLECT_ID];
  if (!posCondCollect) {
    throw new Error('addRegionalGuidance: base graph is missing the positive conditioning collector.');
  }
  const negCondCollect = graph.nodes[NEG_COND_COLLECT_ID] ?? null;
  const support = getRegionalGuidanceSupport(base);
  if (!support) {
    throw new Error(`addRegionalGuidance: ${base} has no regional guidance path.`);
  }

  let ipAdapterCollector: BackendInvocationContract | null = null;
  let fluxReduxCollector: BackendInvocationContract | null = null;

  for (const region of regions) {
    const maskToTensor = addNode(graph, {
      id: `rg_mask_to_tensor_${region.id}`,
      image: { image_name: region.maskImageName },
      type: 'alpha_mask_to_tensor',
    });

    // Positive prompt → positive collector (mask-scoped).
    if (region.positivePrompt) {
      const posCond = addRegionalConditioning(
        graph,
        base,
        modelVariant,
        `rg_pos_cond_${region.id}`,
        region.positivePrompt,
        POS_COND_ID
      );
      addEdge(graph, maskToTensor, 'mask', posCond, 'mask');
      const conditioningSource = transformRegionalPositiveConditioning?.(posCond, region.id) ?? posCond;
      addEdge(graph, conditioningSource, 'conditioning', posCondCollect, 'item');
    }

    // Negative prompt → negative collector.
    if (region.negativePrompt && support.negativePrompt && negCondCollect) {
      const negCond = addRegionalConditioning(
        graph,
        base,
        modelVariant,
        `rg_neg_cond_${region.id}`,
        region.negativePrompt,
        NEG_COND_ID
      );
      addEdge(graph, maskToTensor, 'mask', negCond, 'mask');
      addEdge(graph, negCond, 'conditioning', negCondCollect, 'item');
    }

    // Auto-negative encodes positive text over the inverted region mask.
    if (region.autoNegative && region.positivePrompt && support.autoNegative && negCondCollect) {
      const invert = addNode(graph, { id: `rg_invert_mask_${region.id}`, type: 'invert_tensor_mask' });
      addEdge(graph, maskToTensor, 'mask', invert, 'mask');
      const inverted = addRegionalConditioning(
        graph,
        base,
        modelVariant,
        `rg_pos_cond_inverted_${region.id}`,
        region.positivePrompt,
        POS_COND_ID
      );
      addEdge(graph, invert, 'mask', inverted, 'mask');
      addEdge(graph, inverted, 'conditioning', negCondCollect, 'item');
    }

    // Reference images (mask-scoped): ip_adapter on SD, flux_redux on FLUX.
    for (const ref of region.referenceImages) {
      if (ref.type === 'ip_adapter' && support.referenceImages === 'ip_adapter') {
        if (!ipAdapterCollector) {
          ipAdapterCollector = resolveDenoiseCollector(graph, denoise, 'ip_adapter', 'regional_ip_adapter_collector');
        }
        const node = addNode(graph, {
          begin_step_percent: ref.beginEndStepPct[0],
          clip_vision_model: ref.clipVisionModel,
          end_step_percent: ref.beginEndStepPct[1],
          id: `ip_adapter_${ref.id}`,
          image: { image_name: ref.imageName },
          ip_adapter_model: ref.model,
          method: ref.method,
          type: 'ip_adapter',
          weight: ref.weight,
        });
        addEdge(graph, maskToTensor, 'mask', node, 'mask');
        addEdge(graph, node, 'ip_adapter', ipAdapterCollector, 'item');
      } else if (ref.type === 'flux_redux' && support.referenceImages === 'flux_redux') {
        if (!fluxReduxCollector) {
          fluxReduxCollector = resolveDenoiseCollector(
            graph,
            denoise,
            'redux_conditioning',
            'regional_flux_redux_collector'
          );
        }
        const node = addNode(graph, {
          downsampling_factor: ref.settings.downsampling_factor,
          id: `flux_redux_${ref.id}`,
          image: { image_name: ref.imageName },
          redux_model: ref.model,
          type: 'flux_redux',
          weight: ref.settings.weight,
        });
        addEdge(graph, maskToTensor, 'mask', node, 'mask');
        addEdge(graph, node, 'redux_cond', fluxReduxCollector, 'item');
      }
    }
  }
};

/**
 * Omit unsupported content without erasing stored values. Count resolved references; auto-negative requires
 * positive text. Layout bases need a caption, and a text region its visible text.
 */
export const getRegionalGuidanceRejectionReason = (params: {
  layerName: string;
  mainBase: string;
  hasContent: boolean;
  positivePrompt: string | null;
  negativePrompt: string | null;
  referenceImageCount: number;
  layout?: RegionalLayout;
}): string | null => {
  const { hasContent, layerName, mainBase, negativePrompt, positivePrompt, referenceImageCount } = params;

  const support = getRegionalGuidanceSupport(mainBase);
  if (!support) {
    return `Regional guidance "${layerName}" is not supported for the selected base model.`;
  }
  if (!hasContent) {
    return `Regional guidance "${layerName}" has no masked region.`;
  }
  if (support.mode === 'layout') {
    switch (getLayoutRegionIssue(params)) {
      case 'missing_prompt':
        return `Regional guidance "${layerName}" has no prompt to place in the ${support.label} layout.`;
      case 'missing_text':
        return `Regional guidance "${layerName}" is a text region without visible text.`;
      case null:
        return null;
    }
  }
  const contributes =
    Boolean(positivePrompt) ||
    (Boolean(negativePrompt) && support.negativePrompt) ||
    (referenceImageCount > 0 && support.referenceImages !== null);
  if (!contributes) {
    return `Regional guidance "${layerName}" has no prompt or reference image ${support.label} can use.`;
  }
  return null;
};
