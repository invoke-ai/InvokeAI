import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract, CanvasLayerContract } from '@workbench/canvas-engine/api';

import {
  createControlValidationSequence,
  getControlValidationReason,
  isControlKindSupportedForBase,
  type ControlAdapterKind,
  type ControlValidationReason,
} from '@features/generation/graph';
import { hasArchitectureCapabilities } from '@features/generation/settings';
import { getControlDetectionInput, isControlMapCurrent, needsControlDetection } from '@workbench/canvas-engine/api';
import { hasControlLayerContent, isCompositableControlLayer } from '@workbench/canvasLayerContent';

import { resolveControlModel, type ControlModelMemory, type ControlModelResolution } from './controlModelResolution';

/** Everything that can hold a control layer back: a validation reason, or a map detection has not caught up with. */
export type ControlLayerIssueCode = ControlValidationReason | 'detection_pending';

/** A control layer that blocks generation; the shell words it from the locale (see `controlLayerMessages`). */
export interface ControlLayerIssue {
  code: ControlLayerIssueCode;
  layerId: string;
  layerName: string;
}

/** What model resolution needs beyond the layer: remembered picks and the kinds the main model's base runs. */
export interface ControlLayerCheckContext {
  remembered: ControlModelMemory;
  supportsKind: (kind: ControlAdapterKind) => boolean;
}

/**
 * Before capabilities load every kind counts as supported, so resolution still finds models; validation then
 * reports `capabilities_unavailable` instead of a misleading missing model. Without a main model (`base` null) no
 * base restricts kinds either.
 */
export const createControlLayerCheckContext = (
  base: string | null,
  remembered: ControlModelMemory = {}
): ControlLayerCheckContext => ({
  remembered,
  supportsKind: (kind) => base === null || !hasArchitectureCapabilities() || isControlKindSupportedForBase(base, kind),
});

export { hasControlLayerContent } from '@workbench/canvasLayerContent';

type MainModel = { base: string; variant?: string | null };

export interface ControlLayerJudgement {
  resolution: ControlModelResolution;
  reason: ControlLayerIssueCode | null;
  /** A layer with no type and no model warns but contributes nothing, so it never blocks. */
  skip: boolean;
}

const resolveLayer = (
  layer: CanvasControlLayerContract,
  mainModel: MainModel,
  models: readonly ModelConfig[],
  context: ControlLayerCheckContext
): ControlModelResolution =>
  resolveControlModel({
    base: mainModel.base,
    layer,
    models,
    remembered: context.remembered,
    supportsKind: context.supportsKind,
  });

/**
 * Judges control layers in generation order, as the invocation does: per-generation limits count only layers that
 * passed. Invoke readiness, the settings panel and the invocation collector share it.
 */
export const createControlLayerSequence = (
  mainModel: MainModel,
  models: readonly ModelConfig[],
  context: ControlLayerCheckContext = createControlLayerCheckContext(mainModel.base)
): ((layer: CanvasControlLayerContract) => ControlLayerJudgement) => {
  const validate = createControlValidationSequence(mainModel);
  return (layer) => {
    const resolution = resolveLayer(layer, mainModel, models, context);
    // A layer with no type and no model keeps its reason for the panel but is skipped; having no model, it never
    // passes, so it never claims a per-generation slot.
    const reason =
      validate({
        adapterModel: resolution.model,
        beginEndStepPct: layer.adapter.beginEndStepPct,
        kind: resolution.kind,
        unresolvedModel: resolution.unresolved,
        weight: layer.adapter.weight,
      }) ?? (needsControlDetection(layer) ? 'detection_pending' : null);
    return { reason, resolution, skip: resolution.unresolved === 'missing_type' };
  };
};

/**
 * Validate the same enabled, content-bearing layers as createControlLayerCollector; invalid layers do not consume
 * per-generation limits, and layers without a type or model are left out.
 */
export const getBlockingControlLayerIssues = (params: {
  layers: readonly CanvasLayerContract[];
  mainModel: MainModel;
  models: readonly ModelConfig[];
  context?: ControlLayerCheckContext;
}): ControlLayerIssue[] => {
  const { layers, mainModel, models } = params;
  const judge = createControlLayerSequence(mainModel, models, params.context);
  const issues: ControlLayerIssue[] = [];

  for (const layer of layers) {
    if (!isCompositableControlLayer(layer)) {
      continue;
    }
    const { reason, skip } = judge(layer);
    if (reason && !skip) {
      issues.push({ code: reason, layerId: layer.id, layerName: layer.name });
    }
  }
  return issues;
};

/** Ambient warnings include empty layers but omit per-kind limits, which belong in the Invoke tooltip. */
const judgeAttention = (
  layer: CanvasControlLayerContract,
  mainModel: MainModel,
  models: readonly ModelConfig[],
  context: ControlLayerCheckContext
): { resolution: ControlModelResolution; reason: ControlLayerIssueCode | null } => {
  const resolution = resolveLayer(layer, mainModel, models, context);
  const reason =
    getControlValidationReason({
      adapterModel: resolution.model,
      beginEndStepPct: layer.adapter.beginEndStepPct,
      controlLoraIndex: 0,
      flux2ControlIndex: 0,
      kind: resolution.kind,
      mainBase: mainModel.base,
      mainVariant: mainModel.variant ?? undefined,
      qwenImageControlIndex: 0,
      unresolvedModel: resolution.unresolved,
      weight: layer.adapter.weight,
      zImageControlIndex: 0,
    }) ?? (needsControlDetection(layer) ? 'detection_pending' : null);
  return { reason, resolution };
};

/**
 * Where a layer's control map stands. `empty`: no pixels yet. `none`: no type or no detection, so the pixels are the
 * map. `current`: the map matches its detection. `stale`: detection must run again. `no-input`: the map is all the
 * layer has, so it cannot be detected again until its image is replaced.
 */
export type ControlMapState = 'none' | 'current' | 'stale' | 'no-input' | 'empty';

const getControlMapState = (layer: CanvasControlLayerContract): ControlMapState => {
  if (!hasControlLayerContent(layer) && !layer.original) {
    return 'empty';
  }
  if (!layer.controlType || !layer.detection) {
    return 'none';
  }
  if (isControlMapCurrent(layer)) {
    return 'current';
  }
  return getControlDetectionInput(layer) ? 'stale' : 'no-input';
};

/**
 * One layer as the settings panel, the layer row and Invoke present it. With `earlier` (the contributing layers
 * above it) the reason includes per-generation limits, as Invoke judges it; without, it is the ambient warning.
 * Without a main model the layer resolves across every base and only resolution and detection give a reason.
 */
export const describeControlLayer = (
  layer: CanvasControlLayerContract,
  ctx: ControlLayerCheckContext & {
    mainModel: MainModel | null;
    models: readonly ModelConfig[];
    earlier?: readonly CanvasControlLayerContract[];
  }
): { resolution: ControlModelResolution; map: ControlMapState; reason: ControlLayerIssueCode | null } => {
  const context = { remembered: ctx.remembered, supportsKind: ctx.supportsKind };
  let judgement: { resolution: ControlModelResolution; reason: ControlLayerIssueCode | null };
  if (!ctx.mainModel) {
    const resolution = resolveControlModel({ ...context, base: null, layer, models: ctx.models });
    judgement = {
      reason: resolution.unresolved ?? (needsControlDetection(layer) ? 'detection_pending' : null),
      resolution,
    };
  } else if (ctx.earlier) {
    const judge = createControlLayerSequence(ctx.mainModel, ctx.models, context);
    for (const earlier of ctx.earlier) {
      judge(earlier);
    }
    judgement = judge(layer);
  } else {
    judgement = judgeAttention(layer, ctx.mainModel, ctx.models, context);
  }
  return { map: getControlMapState(layer), reason: judgement.reason, resolution: judgement.resolution };
};

type Translate = (key: string, options?: Record<string, unknown>) => string;

/** The one user-facing wording of a reason, shared by the settings panel, the layer row and the Invoke tooltip. */
export const describeControlLayerReason = (t: Translate, code: ControlLayerIssueCode): string =>
  t(`widgets.layers.control.validation.${code}`);

/** A blocking issue as the Invoke tooltip and the failed-invoke notice show it. */
export const describeControlLayerIssue = (t: Translate, issue: Pick<ControlLayerIssue, 'code' | 'layerName'>): string =>
  t('widgets.layers.control.invalidLayer', {
    name: issue.layerName,
    reason: describeControlLayerReason(t, issue.code),
  });
