import type { ModelConfig } from '@features/models';
import type { CanvasControlAdapterContract } from '@workbench/canvas-engine/api';

import {
  areControlAdapterValuesValid,
  getControlModelUnusableReason,
  isControlModelUsableForKind,
  type ControlAdapterKind,
} from '@features/generation/graph';
import { CONTROL_ADAPTER_DEFAULTS, CONTROL_KIND_BASE } from '@workbench/controlAdapters';

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
 * Anima ControlNet-LLLite models installed before `cond_in_channels` was recorded. The picker lists them disabled, so
 * users learn they exist and need re-identifying; they never count as usable models.
 */
export const getReidentifyPendingControlModels = (
  models: readonly ModelConfig[],
  base: string | null,
  kind: ControlAdapterKind
): ModelConfig[] => {
  if (kind !== 'anima_lllite' || base !== CONTROL_KIND_BASE[kind]) {
    return [];
  }
  return models.filter(
    (model) => model.base === base && getControlModelUnusableReason(model, kind) === 'lllite_channels_unknown'
  );
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
