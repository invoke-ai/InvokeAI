import type { ModelConfig } from '@features/models';
import type { CanvasControlAdapterContract } from '@workbench/canvas-engine/api';

import {
  areControlAdapterValuesValid,
  isControlModelUsableForKind,
  type ControlAdapterKind,
} from '@features/generation/graph';
import { CONTROL_ADAPTER_DEFAULTS, CONTROL_KIND_BASE, getDefaultControlAdapterKind } from '@workbench/controlAdapters';

export const getCompatibleControlModels = (
  models: readonly ModelConfig[],
  base: string | null,
  kind: ControlAdapterKind
): ModelConfig[] => {
  const requiredBase = CONTROL_KIND_BASE[kind];
  if (requiredBase && base !== requiredBase) {
    return [];
  }
  return models.filter((model) => isControlModelUsableForKind(model, kind) && (!base || model.base === base));
};

/**
 * The adapter after the user picks another kind. The model survives when the new kind can use it — so an Anima layer
 * saved as ControlNet with an LLLite model keeps it — and so do the weight and step range when they are valid for
 * the new kind; anything else takes the new kind's default.
 */
export const switchControlAdapterKind = (
  adapter: CanvasControlAdapterContract,
  kind: ControlAdapterKind,
  models: readonly ModelConfig[],
  base: string | null
): CanvasControlAdapterContract => {
  const defaults = CONTROL_ADAPTER_DEFAULTS[kind];
  const keepsModel =
    adapter.model !== null &&
    getCompatibleControlModels(models, base, kind).some((candidate) => candidate.key === adapter.model);
  const keepsWeight = areControlAdapterValuesValid(kind, adapter.weight, defaults.beginEndStepPct);
  const keepsSteps = areControlAdapterValuesValid(kind, defaults.weight, adapter.beginEndStepPct);
  return {
    ...defaults,
    beginEndStepPct: keepsSteps
      ? [adapter.beginEndStepPct[0], adapter.beginEndStepPct[1]]
      : [...defaults.beginEndStepPct],
    model: keepsModel ? adapter.model : null,
    weight: keepsWeight ? adapter.weight : defaults.weight,
  };
};

/**
 * The control model a freshly created layer should start with: a union model
 * (the generalist), then tile, then the first compatible model. Null when no
 * compatible model is installed.
 */
export const resolveDefaultControlModel = (
  models: readonly ModelConfig[],
  base: string | null,
  kind: ControlAdapterKind
): string | null => {
  const compatible = getCompatibleControlModels(models, base, kind);
  const preferred =
    compatible.find((model) => model.name.toLowerCase().includes('union')) ??
    compatible.find((model) => model.name.toLowerCase().includes('tile')) ??
    compatible[0];
  return preferred?.key ?? null;
};

/** Default model for the adapter kind the layer factories pick for `base`. */
export const resolveDefaultControlModelForBase = (models: readonly ModelConfig[], base: string | null): string | null =>
  resolveDefaultControlModel(models, base, getDefaultControlAdapterKind(base));
