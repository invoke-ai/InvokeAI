/** Graft separately composited controls into the graph in place. */

import type { SupportedGenerateBase } from '@features/generation/core/baseGenerationPolicies';
import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';

import { addEdge, addNode } from '@features/generation/core/graphBuilder';

import type { ControlAdapterKind, ControlValidationReason } from './controlValidation';

import { getControlValidationReason } from './controlValidation';

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
}

/** Kind support is separate from control-mode/model compatibility. */
export { isControlKindSupportedForBase } from './controlValidation';

/** Options for {@link addControlLayers}. */
export interface AddControlLayersOptions {
  /** The main model base — selects controlnet vs flux_controlnet and support. */
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
  let controlLoraCount = 0;
  let zImageControlCount = 0;

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
    const reason = getControlValidationReason({
      adapterModel: layer.model,
      beginEndStepPct: layer.beginEndStepPct,
      controlLoraIndex: layer.kind === 'control_lora' ? controlLoraCount : 0,
      kind: layer.kind,
      mainBase: base,
      mainVariant: modelVariant,
      weight: layer.weight,
      zImageControlIndex: layer.kind === 'z_image_control' ? zImageControlCount : 0,
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
      controlLoraCount += 1;
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
      zImageControlCount += 1;
    }
  }
};

/** Return null when valid, otherwise the rejection reason. */
export const getControlLayerRejectionReason = (params: {
  layerName: string;
  hasContent: boolean;
  kind: ControlAdapterKind;
  adapterModel: { base: string; type?: string } | null;
  beginEndStepPct: [number, number];
  controlLoraIndex?: number;
  mainBase: string;
  mainVariant?: string;
  weight: number;
  zImageControlIndex?: number;
}): string | null => {
  const {
    adapterModel,
    beginEndStepPct,
    controlLoraIndex = 0,
    hasContent,
    kind,
    layerName,
    mainBase,
    mainVariant,
    weight,
    zImageControlIndex = 0,
  } = params;

  if (!hasContent) {
    return `Control layer "${layerName}" has no control content.`;
  }
  const reason = getControlValidationReason({
    adapterModel: adapterModel ? { base: adapterModel.base, type: adapterModel.type ?? kind } : null,
    beginEndStepPct,
    controlLoraIndex,
    kind,
    mainBase,
    mainVariant,
    weight,
    zImageControlIndex,
  });
  return reason ? getControlValidationReasonMessage(reason, layerName) : null;
};

/** The human-readable sentence for a control validation reason code. */
export const getControlValidationReasonMessage = (reason: ControlValidationReason, layerName: string): string => {
  if (reason === 'missing_model') {
    return `Control layer "${layerName}" has no control model selected.`;
  }
  if (reason === 'capabilities_unavailable') {
    return `Control layer "${layerName}" cannot be checked until model capabilities have loaded.`;
  }
  if (reason === 'unsupported_adapter') {
    return `Control layer "${layerName}" is not supported for the selected base model.`;
  }
  if (reason === 'incompatible_base') {
    return `Control layer "${layerName}" uses an incompatible base model.`;
  }
  if (reason === 'invalid_adapter_values') {
    return `Control layer "${layerName}" has invalid control adapter settings.`;
  }
  if (reason === 'control_lora_limit') {
    return 'Only one Control LoRA can be used at a time.';
  }
  if (reason === 'z_image_control_limit') {
    return 'Only one Z-Image control layer can be used at a time.';
  }
  if (reason === 'flux_fill_control_lora') {
    return 'Control LoRA is not compatible with FLUX Fill.';
  }
  return `Control layer "${layerName}" uses an incompatible control adapter.`;
};
