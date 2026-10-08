/** Graft separately composited controls into the graph in place. */

import type { SupportedGenerateBase } from '@features/generation/core/baseGenerationPolicies';
import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';

import { addEdge, addNode } from '@features/generation/core/graphBuilder';

import type { ControlAdapterKind } from './controlValidation';

import { createControlValidationSequence } from './controlValidation';

/** The deterministic denoise node id every canvas base graph uses. */
export const CONTROL_DENOISE_NODE_ID = 'denoise_latents';

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

/** Options for {@link addControlLayers}. */
export interface AddControlLayersOptions {
  /** The main model base — selects the adapter nodes and support. */
  base: SupportedGenerateBase;
  /** The main model variant (a FLUX `dev_fill` blocks Control LoRA). */
  modelVariant?: string;
  /** The valid, resolved control layers to graft (in document order). */
  layers: readonly ControlLayerGraphInput[];
}

/** Resolves the backend node type for a controlnet layer on `base`. */
const controlNetNodeType = (base: string): string => (base === 'flux' ? 'flux_controlnet' : 'controlnet');

/** Validate defensively through the shared policy before graph wiring. */
export const addControlLayers = (graph: BackendGraphContract, options: AddControlLayersOptions): void => {
  const { base, layers, modelVariant } = options;
  const denoise = graph.nodes[CONTROL_DENOISE_NODE_ID];
  if (!denoise) {
    throw new Error('addControlLayers: base graph is missing the denoise node.');
  }

  let controlNetCollector: BackendInvocationContract | null = null;
  let t2iAdapterCollector: BackendInvocationContract | null = null;
  let lliteCollector: BackendInvocationContract | null = null;
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

  // `anima_denoise.control_lllite` takes one field or a list; a collector carries any count through one edge.
  const ensureLliteCollector = (): BackendInvocationContract => {
    if (!lliteCollector) {
      lliteCollector = addNode(graph, { id: 'control_lllite_collector', type: 'collect' });
      addEdge(graph, lliteCollector, 'collection', denoise, 'control_lllite');
    }
    return lliteCollector;
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
      addEdge(graph, node, 'control', ensureLliteCollector(), 'item');
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
