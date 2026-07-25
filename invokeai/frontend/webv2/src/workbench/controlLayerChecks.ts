import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract, CanvasLayerContract } from '@workbench/canvas-engine/api';

import {
  getControlLayerRejectionReason,
  getControlValidationReason,
  type ControlValidationReason,
} from '@features/generation/graph';

export interface ControlLayerIssue {
  code: ControlValidationReason;
  layerId: string;
  layerName: string;
  /** Human-readable sentence (plain English, matching other invocation reasons). */
  message: string;
}

/**
 * Pure parity with the composite plan's content gate: a control layer only
 * reaches the generation pipeline when it has an image source or committed
 * paint pixels, so readiness checks must skip anything else.
 */
export const hasControlLayerContent = (layer: CanvasControlLayerContract): boolean => {
  if (layer.source.type === 'image') {
    return true;
  }
  return layer.source.type === 'paint' && layer.source.bitmap !== null;
};

const resolveAdapterModel = (
  adapterModel: string | null,
  models: readonly ModelConfig[]
): { base: string; type: string } | null => {
  const resolved = adapterModel ? models.find((candidate) => candidate.key === adapterModel) : undefined;
  return resolved ? { base: resolved.base, type: resolved.type } : null;
};

/**
 * The control-layer problems that would make the invoke pipeline reject the
 * document, in document order. Mirrors `createControlLayerCollector`: only
 * enabled, content-bearing control layers are validated, and the per-kind
 * limit counters advance only past valid layers.
 */
export const getBlockingControlLayerIssues = (params: {
  layers: readonly CanvasLayerContract[];
  mainModel: { base: string; variant?: string | null };
  models: readonly ModelConfig[];
}): ControlLayerIssue[] => {
  const { layers, mainModel, models } = params;
  const issues: ControlLayerIssue[] = [];
  let controlLoraCount = 0;
  let zImageControlCount = 0;

  for (const layer of layers) {
    if (layer.type !== 'control' || !layer.isEnabled || !hasControlLayerContent(layer)) {
      continue;
    }
    const { adapter } = layer;
    const adapterModel = resolveAdapterModel(adapter.model, models);
    const controlLoraIndex = adapter.kind === 'control_lora' ? controlLoraCount : 0;
    const zImageControlIndex = adapter.kind === 'z_image_control' ? zImageControlCount : 0;
    const code = getControlValidationReason({
      adapterModel,
      beginEndStepPct: adapter.beginEndStepPct,
      controlLoraIndex,
      kind: adapter.kind,
      mainBase: mainModel.base,
      mainVariant: mainModel.variant ?? undefined,
      weight: adapter.weight,
      zImageControlIndex,
    });
    if (code) {
      const message =
        getControlLayerRejectionReason({
          adapterModel,
          beginEndStepPct: adapter.beginEndStepPct,
          controlLoraIndex,
          hasContent: true,
          kind: adapter.kind,
          layerName: layer.name,
          mainBase: mainModel.base,
          mainVariant: mainModel.variant ?? undefined,
          weight: adapter.weight,
          zImageControlIndex,
        }) ?? `Control layer "${layer.name}" is invalid.`;
      issues.push({ code, layerId: layer.id, layerName: layer.name, message });
      continue;
    }
    if (adapter.kind === 'control_lora') {
      controlLoraCount += 1;
    }
    if (adapter.kind === 'z_image_control') {
      zImageControlCount += 1;
    }
  }
  return issues;
};

/**
 * Relaxed per-layer check for ambient warning UI: content is not required
 * (a fresh model-less layer should already warn) and the per-kind limit
 * counters are pinned to zero (limit problems stay in the invoke tooltip).
 */
export const getControlLayerAttentionReason = (
  layer: CanvasControlLayerContract,
  mainBase: string,
  models: readonly ModelConfig[]
): ControlValidationReason | null =>
  getControlValidationReason({
    adapterModel: resolveAdapterModel(layer.adapter.model, models),
    beginEndStepPct: layer.adapter.beginEndStepPct,
    controlLoraIndex: 0,
    kind: layer.adapter.kind,
    mainBase,
    weight: layer.adapter.weight,
    zImageControlIndex: 0,
  });
