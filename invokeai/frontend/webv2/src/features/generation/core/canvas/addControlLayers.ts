/** Graft separately composited controls into the graph in place. */

import type { SupportedGenerateBase } from '@features/generation/core/baseGenerationPolicies';
import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';

import { addEdge, addNode } from '@features/generation/core/graphBuilder';

import type { ControlAdapterKind } from './controlValidation';

import { createControlValidationSequence } from './controlValidation';

/** The deterministic denoise node id every canvas base graph uses. */
export const CONTROL_DENOISE_NODE_ID = 'denoise_latents';

const LLLITE_COLLECTOR_NODE_ID = 'control_lllite_collector';

/** The control-adapter kinds a control layer can carry. */
export type { ControlAdapterKind } from './controlValidation';

/** A resolved control-adapter model identifier (the backend model field shape). */
export interface ControlModelIdentifier {
  key: string;
  hash?: string;
  name: string;
  base: string;
  type: string;
}

/** One control layer's fully-resolved graph contribution (invalid layers filtered out first). */
export interface ControlLayerGraphInput {
  /** The document layer id (used to mint deterministic node ids). */
  id: string;
  /** The uploaded per-layer composite image name. */
  imageName: string;
  kind: ControlAdapterKind;
  /** The resolved control-adapter model identifier (never null here). */
  model: ControlModelIdentifier;
  weight: number;
  beginEndStepPct: [number, number];
  controlMode: 'balanced' | 'more_prompt' | 'more_control' | 'unbalanced' | null;
  /** The resolved model's `cond_in_channels`; required to validate an Anima ControlNet-LLLite layer. */
  modelCondInChannels?: number | null;
}

/** Kind support is separate from control-mode/model compatibility. */
export { isControlKindSupportedForBase } from './controlValidation';

/** A node feeding an image input through an edge, or a fixed image field when no node produces it. */
export type ImageSource = { node: BackendInvocationContract; field: string } | { image_name: string };

/**
 * What the canvas inpaint and outpaint graphs regenerate: the image the denoise encodes, at processing size, and
 * the canvas mask as uploaded (white = keep). Adapters that condition on the region read both.
 */
export interface CanvasInpaintSources {
  image: ImageSource;
  mask: ImageSource;
}

/** Feeds `field` of `node` from `source`: an edge from the producing node, or the fixed image field. */
const feedImage = (
  graph: BackendGraphContract,
  node: BackendInvocationContract,
  field: string,
  source: ImageSource
): void => {
  if ('node' in source) {
    addEdge(graph, source.node, source.field, node, field);
  } else {
    node[field] = source;
  }
};

/** Options for {@link addControlLayers}. */
export interface AddControlLayersOptions {
  /** The main model base — selects the adapter nodes and support. */
  base: SupportedGenerateBase;
  /** The main model variant (a FLUX `dev_fill` blocks Control LoRA; a Qwen-Image `edit` blocks its control). */
  modelVariant?: string;
  /** The valid, resolved control layers to graft (in document order). */
  layers: readonly ControlLayerGraphInput[];
  /**
   * Present when the canvas compiles in inpaint or outpaint mode. The Qwen-Image control node also conditions on
   * the inpainted region and takes them; the other kinds ignore them.
   */
  inpaint?: CanvasInpaintSources;
}

/**
 * The one collector every Anima ControlNet-LLLite adapter joins. `anima_denoise.control_lllite` takes one field or a
 * list, and the denoiser applies each model once, so control layers and the canvas inpaint adapter share it.
 */
export const getOrAddLliteCollector = (
  graph: BackendGraphContract,
  denoise: BackendInvocationContract
): BackendInvocationContract => {
  const existing = graph.nodes[LLLITE_COLLECTOR_NODE_ID];
  if (existing) {
    return existing;
  }
  const collector = addNode(graph, { id: LLLITE_COLLECTOR_NODE_ID, type: 'collect' });
  addEdge(graph, collector, 'collection', denoise, 'control_lllite');
  return collector;
};

/** Resolves the backend node type for a controlnet layer on `base`. */
const controlNetNodeType = (base: string): string => (base === 'flux' ? 'flux_controlnet' : 'controlnet');

/** Validate defensively through the shared policy before graph wiring. */
export const addControlLayers = (graph: BackendGraphContract, options: AddControlLayersOptions): void => {
  const { base, inpaint, layers, modelVariant } = options;
  const denoise = graph.nodes[CONTROL_DENOISE_NODE_ID];
  if (!denoise) {
    throw new Error('addControlLayers: base graph is missing the denoise node.');
  }

  let controlNetCollector: BackendInvocationContract | null = null;
  let t2iAdapterCollector: BackendInvocationContract | null = null;
  const validate = createControlValidationSequence({ base, variant: modelVariant });

  const ensureControlNetCollector = (): BackendInvocationContract => {
    if (!controlNetCollector) {
      controlNetCollector = addNode(graph, { id: 'control_net_collector', type: 'collect' });
      addEdge(graph, controlNetCollector, 'collection', denoise, 'control');
    }
    return controlNetCollector;
  };

  const ensureT2iAdapterCollector = (): BackendInvocationContract => {
    if (!t2iAdapterCollector) {
      t2iAdapterCollector = addNode(graph, { id: 't2i_adapter_collector', type: 'collect' });
      addEdge(graph, t2iAdapterCollector, 'collection', denoise, 't2i_adapter');
    }
    return t2iAdapterCollector;
  };

  for (const layer of layers) {
    const reason = validate({
      adapterModel: { ...layer.model, cond_in_channels: layer.modelCondInChannels },
      beginEndStepPct: layer.beginEndStepPct,
      kind: layer.kind,
      weight: layer.weight,
    });
    if (reason) {
      throw new Error(`Invalid control layer: ${reason}`);
    }

    if (layer.kind === 'controlnet') {
      const node = addNode(graph, {
        begin_step_percent: layer.beginEndStepPct[0],
        control_model: layer.model,
        control_weight: layer.weight,
        end_step_percent: layer.beginEndStepPct[1],
        id: `control_net_${layer.id}`,
        image: { image_name: layer.imageName },
        resize_mode: 'just_resize',
        type: controlNetNodeType(base),
        // FLUX ControlNet has no control_mode; SD-family carries it.
        ...(base === 'flux' ? {} : { control_mode: layer.controlMode ?? 'balanced' }),
      });
      addEdge(graph, node, 'control', ensureControlNetCollector(), 'item');
    } else if (layer.kind === 't2i_adapter') {
      const node = addNode(graph, {
        begin_step_percent: layer.beginEndStepPct[0],
        end_step_percent: layer.beginEndStepPct[1],
        id: `t2i_adapter_${layer.id}`,
        image: { image_name: layer.imageName },
        resize_mode: 'just_resize',
        t2i_adapter_model: layer.model,
        type: 't2i_adapter',
        weight: layer.weight,
      });
      addEdge(graph, node, 't2i_adapter', ensureT2iAdapterCollector(), 'item');
    } else if (layer.kind === 'control_lora') {
      const node = addNode(graph, {
        id: `control_lora_${layer.id}`,
        image: { image_name: layer.imageName },
        lora: layer.model,
        type: 'flux_control_lora_loader',
        weight: layer.weight,
      });
      addEdge(graph, node, 'control_lora', denoise, 'control_lora');
    } else if (layer.kind === 'anima_lllite') {
      // No mask: a control layer supplies a control image, and validation admits only 3-channel adapters.
      const node = addNode(graph, {
        begin_step_percent: layer.beginEndStepPct[0],
        control_model: layer.model,
        end_step_percent: layer.beginEndStepPct[1],
        id: `anima_lllite_${layer.id}`,
        image: { image_name: layer.imageName },
        type: 'anima_lllite',
        weight: layer.weight,
      });
      addEdge(graph, node, 'control', getOrAddLliteCollector(graph, denoise), 'item');
    } else if (layer.kind === 'qwen_image_control') {
      // One field on the denoise, no collector: validation admits a single Qwen-Image control per generation.
      const node = addNode(graph, {
        begin_step_percent: layer.beginEndStepPct[0],
        control_context_scale: layer.weight,
        control_model: layer.model,
        end_step_percent: layer.beginEndStepPct[1],
        id: `qwen_image_control_${layer.id}`,
        image: { image_name: layer.imageName },
        type: 'qwen_image_control',
      });
      if (inpaint) {
        // The union's inpainting condition. The canvas mask is white = keep and the node wants white = regenerate,
        // so `img_lerp` 255 → 0 inverts it first, as the Anima inpaint adapter does.
        const invertMask = addNode(graph, {
          id: `qwen_image_control_mask_${layer.id}`,
          max: 0,
          min: 255,
          type: 'img_lerp',
        });
        feedImage(graph, invertMask, 'image', inpaint.mask);
        feedImage(graph, node, 'inpaint_image', inpaint.image);
        addEdge(graph, invertMask, 'image', node, 'inpaint_mask');
      }
      addEdge(graph, node, 'control', denoise, 'control');
    } else {
      const node = addNode(graph, {
        begin_step_percent: layer.beginEndStepPct[0],
        control_context_scale: layer.weight,
        control_model: layer.model,
        end_step_percent: layer.beginEndStepPct[1],
        id: `z_image_control_${layer.id}`,
        image: { image_name: layer.imageName },
        type: 'z_image_control',
      });
      addEdge(graph, node, 'control', denoise, 'control');
    }
  }
};
