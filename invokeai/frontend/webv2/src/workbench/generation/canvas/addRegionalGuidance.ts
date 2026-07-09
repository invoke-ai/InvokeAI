/**
 * Pure regional-guidance graph grafting (legacy parity).
 *
 * {@link addRegionalGuidance} wires each enabled, valid regional-guidance region
 * into the canvas base graph's conditioning collectors: a per-region
 * `alpha_mask_to_tensor` node (fed by the region's uploaded mask image) supplies
 * a mask tensor that scopes per-region positive/negative prompt conditioning
 * (into `pos_cond_collect` / `neg_cond_collect`), an `autoNegative` inverted-mask
 * negative conditioning, and mask-scoped reference images (SD `ip_adapter`,
 * FLUX `flux_redux`). It mutates the graph in place using the same
 * `addNode`/`addEdge` helpers the base builders use — no fetch, no engine, no
 * React. Node types + args mirror legacy
 * `features/nodes/util/graph/generation/addRegions.ts`.
 *
 * ## Support matrix (legacy `getRegionalGuidanceWarnings`)
 *
 * - **sd-1 / sdxl**: full support — positive, negative, autoNegative, regional
 *   `ip_adapter` reference images.
 * - **flux**: positive prompt + `flux_redux` reference images only. Negative
 *   prompt, autoNegative, and regional `ip_adapter` are NOT supported (silently
 *   skipped here; surfaced as a rejection reason upstream).
 * - Every other base is unsupported (region skipped; rejection reason emitted).
 *
 * The caller (`prepareCanvasInvocation`) composites + uploads each region's mask
 * separately and resolves reference-image models, then passes only valid regions
 * in — this module only shapes nodes/edges.
 */

import type { BackendGraphContract, BackendInvocationContract } from '@workbench/types';

import { addEdge, addNode } from '@workbench/generation/graphBuilder';

/** The deterministic denoise node id every canvas base graph uses. */
const DENOISE_NODE_ID = 'denoise_latents';
/** The deterministic global positive/negative conditioning + collector node ids. */
const POS_COND_ID = 'pos_cond';
const NEG_COND_ID = 'neg_cond';
const POS_COND_COLLECT_ID = 'pos_cond_collect';
const NEG_COND_COLLECT_ID = 'neg_cond_collect';

/** The base models regional guidance supports (legacy: sd-1 / sdxl / flux). */
export type RegionalGuidanceBase = 'sd-1' | 'sdxl' | 'flux';

/** True when `base` supports regional guidance at all (SD1 / SDXL / FLUX). */
export const isRegionalGuidanceSupportedForBase = (base: string): base is RegionalGuidanceBase =>
  base === 'sd-1' || base === 'sdxl' || base === 'flux';

/** Whether a base supports regional NEGATIVE prompts / autoNegative (SD family only, not FLUX). */
const supportsRegionalNegative = (base: RegionalGuidanceBase): boolean => base === 'sd-1' || base === 'sdxl';

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
  regions: readonly RegionalGuidanceInput[];
}

/** Per-base conditioning encoder node type + the fields carrying the prompt. */
const conditioningNodeType = (base: RegionalGuidanceBase): string => {
  switch (base) {
    case 'sdxl':
      return 'sdxl_compel_prompt';
    case 'flux':
      return 'flux_text_encoder';
    case 'sd-1':
      return 'compel';
  }
};

/** The prompt input fields to set on a regional conditioning node (SDXL mirrors prompt→style). */
const promptFields = (base: RegionalGuidanceBase): readonly string[] =>
  base === 'sdxl' ? ['prompt', 'style'] : ['prompt'];

/**
 * The encoder-input fields to COPY from the global conditioning node onto a
 * regional one, so the region shares the same CLIP / T5 encoders (legacy copies
 * the CLIP/T5 edges verbatim). `mask` is wired separately, so it's excluded here.
 */
const copyEncoderFields = (base: RegionalGuidanceBase): readonly string[] => {
  switch (base) {
    case 'sdxl':
      return ['clip', 'clip2'];
    case 'flux':
      return ['clip', 't5_encoder', 't5_max_seq_len'];
    case 'sd-1':
      return ['clip'];
  }
};

/**
 * Copies every edge feeding `sourceNodeId`'s `fields` onto `target` (same source,
 * same field). Used to share the global conditioning node's CLIP/T5 encoder
 * inputs with a per-region conditioning node.
 */
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
  nodeId: string,
  prompt: string,
  copyFrom: string
): BackendInvocationContract => {
  const node = addNode(graph, { id: nodeId, type: conditioningNodeType(base) });
  for (const field of promptFields(base)) {
    (node as Record<string, unknown>)[field] = prompt;
  }
  copyEncoderEdges(graph, copyFrom, node, copyEncoderFields(base));
  return node;
};

/**
 * Grafts regional guidance onto a built canvas base graph. Assumes every input is
 * already validated for `base` (supported base, non-empty region, resolved
 * reference-image models) — use {@link getRegionalGuidanceRejectionReason} to
 * filter first. Wires, per enabled region:
 * - `alpha_mask_to_tensor` from the uploaded region mask;
 * - positive prompt → regional conditioning → `pos_cond_collect`;
 * - negative prompt (SD only) → regional conditioning → `neg_cond_collect`;
 * - autoNegative (SD only) → `invert_tensor_mask` + positive prompt re-encoded →
 *   `neg_cond_collect` (push the positive prompt away outside the region);
 * - reference images → mask-scoped `ip_adapter` (SD) / `flux_redux` (FLUX).
 */
export const addRegionalGuidance = (graph: BackendGraphContract, options: AddRegionalGuidanceOptions): void => {
  const { base, regions } = options;
  const denoise = graph.nodes[DENOISE_NODE_ID];
  if (!denoise) {
    throw new Error('addRegionalGuidance: base graph is missing the denoise node.');
  }
  const posCondCollect = graph.nodes[POS_COND_COLLECT_ID];
  if (!posCondCollect) {
    throw new Error('addRegionalGuidance: base graph is missing the positive conditioning collector.');
  }
  const negCondCollect = graph.nodes[NEG_COND_COLLECT_ID] ?? null;
  const withNegative = supportsRegionalNegative(base);

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
        `rg_pos_cond_${region.id}`,
        region.positivePrompt,
        POS_COND_ID
      );
      addEdge(graph, maskToTensor, 'mask', posCond, 'mask');
      addEdge(graph, posCond, 'conditioning', posCondCollect, 'item');
    }

    // Negative prompt → negative collector (SD only; FLUX has no negative path).
    if (region.negativePrompt && withNegative && negCondCollect) {
      const negCond = addRegionalConditioning(
        graph,
        base,
        `rg_neg_cond_${region.id}`,
        region.negativePrompt,
        NEG_COND_ID
      );
      addEdge(graph, maskToTensor, 'mask', negCond, 'mask');
      addEdge(graph, negCond, 'conditioning', negCondCollect, 'item');
    }

    // autoNegative: re-encode the POSITIVE prompt over the INVERTED mask into the
    // negative collector — pushes the region's prompt away everywhere outside it.
    if (region.autoNegative && region.positivePrompt && withNegative && negCondCollect) {
      const invert = addNode(graph, { id: `rg_invert_mask_${region.id}`, type: 'invert_tensor_mask' });
      addEdge(graph, maskToTensor, 'mask', invert, 'mask');
      const inverted = addRegionalConditioning(
        graph,
        base,
        `rg_pos_cond_inverted_${region.id}`,
        region.positivePrompt,
        POS_COND_ID
      );
      addEdge(graph, invert, 'mask', inverted, 'mask');
      addEdge(graph, inverted, 'conditioning', negCondCollect, 'item');
    }

    // Reference images (mask-scoped): ip_adapter on SD, flux_redux on FLUX.
    for (const ref of region.referenceImages) {
      if (ref.type === 'ip_adapter' && base !== 'flux') {
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
      } else if (ref.type === 'flux_redux' && base === 'flux') {
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
 * Returns the legacy-parity rejection reason for a regional-guidance region, or
 * `null` when it can contribute to generation. Mirrors
 * `getRegionalGuidanceWarnings`:
 * - unsupported main base (sd-2 / sd-3 / cogview / …) → "unsupported model";
 * - no drawn mask content → "no region";
 * - no positive prompt, no negative prompt, and no reference images → "empty";
 * - FLUX with a negative prompt / autoNegative → those are unsupported on FLUX.
 *
 * Per-reference-image model/image validity is resolved by the caller (which drops
 * incomplete reference images before building), matching how control layers work.
 */
export const getRegionalGuidanceRejectionReason = (params: {
  layerName: string;
  mainBase: string;
  hasContent: boolean;
  positivePrompt: string | null;
  negativePrompt: string | null;
  autoNegative: boolean;
  referenceImageCount: number;
}): string | null => {
  const { autoNegative, hasContent, layerName, mainBase, negativePrompt, positivePrompt, referenceImageCount } = params;

  if (!isRegionalGuidanceSupportedForBase(mainBase)) {
    return `Regional guidance "${layerName}" is not supported for the selected base model.`;
  }
  if (!hasContent) {
    return `Regional guidance "${layerName}" has no masked region.`;
  }
  if (!positivePrompt && !negativePrompt && referenceImageCount === 0) {
    return `Regional guidance "${layerName}" has no prompt or reference image.`;
  }
  if (mainBase === 'flux' && negativePrompt) {
    return `Regional guidance "${layerName}" negative prompts are not supported for FLUX.`;
  }
  if (mainBase === 'flux' && autoNegative) {
    return `Regional guidance "${layerName}" auto-negative is not supported for FLUX.`;
  }
  return null;
};
