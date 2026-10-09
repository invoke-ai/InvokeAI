import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract, CanvasLayerContract } from '@workbench/canvas-engine/api';

import {
  createControlValidationSequence,
  getControlValidationReason,
  getSuggestedControlKind,
  type ControlAdapterKind,
  type ControlValidationReason,
} from '@features/generation/graph';
import { isCompositableControlLayer } from '@workbench/canvasLayerContent';

/** A control layer that blocks generation; the shell words it from the locale (see `controlLayerMessages`). */
export interface ControlLayerIssue {
  code: ControlValidationReason;
  layerId: string;
  layerName: string;
  /** For `switch_adapter_kind`: the kind the selected base runs. */
  suggestedKind: ControlAdapterKind | null;
}

export { hasControlLayerContent } from '@workbench/canvasLayerContent';

const resolveAdapterModel = (adapterModel: string | null, models: readonly ModelConfig[]): ModelConfig | null =>
  (adapterModel ? models.find((candidate) => candidate.key === adapterModel) : undefined) ?? null;

type MainModel = { base: string; variant?: string | null };

/** Validates control layers in generation order, as the invocation does. */
const createLayerValidator = (mainModel: MainModel, models: readonly ModelConfig[]) => {
  const validate = createControlValidationSequence(mainModel);
  return ({ adapter }: CanvasControlLayerContract): ControlValidationReason | null =>
    validate({
      adapterModel: resolveAdapterModel(adapter.model, models),
      beginEndStepPct: adapter.beginEndStepPct,
      kind: adapter.kind,
      weight: adapter.weight,
    });
};

/**
 * Validate the same enabled, content-bearing layers as createControlLayerCollector; invalid layers do not consume
 * per-generation limits.
 */
export const getBlockingControlLayerIssues = (params: {
  layers: readonly CanvasLayerContract[];
  mainModel: MainModel;
  models: readonly ModelConfig[];
}): ControlLayerIssue[] => {
  const { layers, mainModel, models } = params;
  const validate = createLayerValidator(mainModel, models);
  const issues: ControlLayerIssue[] = [];

  for (const layer of layers) {
    if (!isCompositableControlLayer(layer)) {
      continue;
    }
    const code = validate(layer);
    if (code) {
      issues.push({
        code,
        layerId: layer.id,
        layerName: layer.name,
        suggestedKind: code === 'switch_adapter_kind' ? getSuggestedControlKind(mainModel.base) : null,
      });
    }
  }
  return issues;
};

/**
 * The reason one layer would block generation, judged exactly as the invocation judges it: `earlier` are the
 * contributing, content-bearing control layers above it in generation order, which may claim limited slots first.
 */
export const getControlLayerReasonInSequence = (params: {
  earlier: readonly CanvasControlLayerContract[];
  layer: CanvasControlLayerContract;
  mainModel: MainModel;
  models: readonly ModelConfig[];
}): ControlValidationReason | null => {
  const validate = createLayerValidator(params.mainModel, params.models);
  for (const earlier of params.earlier) {
    validate(earlier);
  }
  return validate(params.layer);
};

/** Ambient warnings include empty layers but omit per-kind limits, which belong in the Invoke tooltip. */
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

type Translate = (key: string, options?: Record<string, unknown>) => string;

/** The one user-facing wording of a reason, shared by the settings panel, the layer row and the Invoke tooltip. */
export const describeControlLayerReason = (
  t: Translate,
  code: ControlValidationReason,
  suggestedKind: ControlAdapterKind | null
): string =>
  t(`widgets.layers.control.validation.${code}`, {
    kind: suggestedKind ? t(`widgets.layers.control.kinds.${suggestedKind}`) : '',
  });

/** A blocking issue as the Invoke tooltip and the failed-invoke notice show it. */
export const describeControlLayerIssue = (
  t: Translate,
  issue: Pick<ControlLayerIssue, 'code' | 'layerName' | 'suggestedKind'>
): string =>
  t('widgets.layers.control.invalidLayer', {
    name: issue.layerName,
    reason: describeControlLayerReason(t, issue.code, issue.suggestedKind),
  });
