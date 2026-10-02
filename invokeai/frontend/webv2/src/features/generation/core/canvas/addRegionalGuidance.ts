import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';

import { getArchitectureFeatures } from '@features/generation/core/architectureCapabilities';
import { addEdge, addNode } from '@features/generation/core/graphBuilder';

/** The deterministic denoise node id every canvas base graph uses. */
const DENOISE_NODE_ID = 'denoise_latents';
/** The deterministic global positive/negative conditioning + collector node ids. */
const POS_COND_ID = 'pos_cond';
const NEG_COND_ID = 'neg_cond';
const POS_COND_COLLECT_ID = 'pos_cond_collect';
const NEG_COND_COLLECT_ID = 'neg_cond_collect';

/** The base models whose text encoder accepts a region mask and whose denoiser accepts a conditioning list. */
export type RegionalGuidanceBase = 'sd-1' | 'sd-2' | 'sdxl' | 'flux' | 'flux2' | 'krea-2' | 'z-image' | 'anima';

/** The reference-image node kind a base's regions can consume. */
export type RegionalReferenceImageKind = 'ip_adapter' | 'flux_redux';

/** What the backend honours per region for one base, beyond the masked positive prompt. */
export interface RegionalGuidanceSupport {
  /** Display name used in rejection reasons and settings hints. */
  label: string;
  /** Masked regional negative prompts reach the negative collector. */
  negativePrompt: boolean;
  /** The positive prompt can be re-encoded over the inverted mask as a negative. */
  autoNegative: boolean;
  /** Mask-scoped reference images, or `null` when the denoiser has no regional image path. */
  referenceImages: RegionalReferenceImageKind | null;
}

/** Graph code owns node mapping; the backend owns regional-negative support. */
const REGIONAL_GUIDANCE_SUPPORT: Record<
  RegionalGuidanceBase,
  Pick<RegionalGuidanceSupport, 'label' | 'referenceImages'>
> = {
  'sd-1': { label: 'SD 1.x', referenceImages: 'ip_adapter' },
  'sd-2': { label: 'SD 2.x', referenceImages: 'ip_adapter' },
  sdxl: { label: 'SDXL', referenceImages: 'ip_adapter' },
  flux: { label: 'FLUX', referenceImages: 'flux_redux' },
  flux2: { label: 'FLUX.2', referenceImages: null },
  'krea-2': { label: 'Krea-2', referenceImages: null },
  'z-image': { label: 'Z-Image', referenceImages: null },
  anima: { label: 'Anima', referenceImages: null },
};

/** Frontend graph keys are narrower than backend capability answers. */
export const isRegionalGuidanceSupportedForBase = (base: string): base is RegionalGuidanceBase =>
  // Require both backend support and a local graph row before narrowing the type.
  (getArchitectureFeatures(base)?.supports_regional_guidance ?? false) && base in REGIONAL_GUIDANCE_SUPPORT;

/** Auto-negative requires negative-mask support. */
export const getRegionalGuidanceSupport = (base: string | null): RegionalGuidanceSupport | null => {
  if (base === null || !isRegionalGuidanceSupportedForBase(base)) {
    return null;
  }

  const negative = getArchitectureFeatures(base)?.regional_negative ?? false;

  return { ...REGIONAL_GUIDANCE_SUPPORT[base], autoNegative: negative, negativePrompt: negative };
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
}

/** Options for {@link addRegionalGuidance}. */
export interface AddRegionalGuidanceOptions {
  base: RegionalGuidanceBase;
  modelVariant?: string | null;
  regions: readonly RegionalGuidanceInput[];
  /** Applies model-specific transforms before a regional positive conditioning is collected. */
  transformRegionalPositiveConditioning?: (
    conditioning: BackendInvocationContract,
    regionId: string
  ) => BackendInvocationContract;
}

/** Per-base conditioning encoder node type + the fields carrying the prompt. */
const conditioningNodeType = (base: RegionalGuidanceBase, modelVariant?: string | null): string => {
  switch (base) {
    case 'sdxl':
      return 'sdxl_compel_prompt';
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
    case 'sd-1':
    case 'sd-2':
      return 'compel';
  }
};

/** The prompt input fields to set on a regional conditioning node (SDXL mirrors prompt→style). */
const promptFields = (base: RegionalGuidanceBase): readonly string[] =>
  base === 'sdxl' ? ['prompt', 'style'] : ['prompt'];

/** Copy global encoder fields; wire masks separately. */
const copyEncoderFields = (base: RegionalGuidanceBase, modelVariant?: string | null): readonly string[] => {
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
    case 'anima':
      return ['qwen3_encoder'];
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
  base: RegionalGuidanceBase,
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

/** Inputs must be prevalidated and gated by regional support. */
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
 * positive text.
 */
export const getRegionalGuidanceRejectionReason = (params: {
  layerName: string;
  mainBase: string;
  hasContent: boolean;
  positivePrompt: string | null;
  negativePrompt: string | null;
  referenceImageCount: number;
}): string | null => {
  const { hasContent, layerName, mainBase, negativePrompt, positivePrompt, referenceImageCount } = params;

  const support = getRegionalGuidanceSupport(mainBase);
  if (!support) {
    return `Regional guidance "${layerName}" is not supported for the selected base model.`;
  }
  if (!hasContent) {
    return `Regional guidance "${layerName}" has no masked region.`;
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
