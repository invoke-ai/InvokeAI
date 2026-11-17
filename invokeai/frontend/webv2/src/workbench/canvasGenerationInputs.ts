import type { GenerateModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/api';

import {
  FLUX_REDUX_INFLUENCE,
  getLayoutPromptRejectionReason,
  getRegionalGuidanceRejectionReason,
  getRegionalGuidanceSupport,
  type ControlLayerGraphInput,
  type RegionalGuidanceInput,
  type RegionalReferenceImageInput,
} from '@features/generation/graph';
import { describeExternalCanvasRefusal, isExternalTextOnlyCanvasModel } from '@features/generation/settings';
import { isCompositableControlLayer, isCompositableRegionalGuidanceLayer } from '@workbench/canvasLayerContent';

import type { ControlModelMemory } from './controlModelResolution';

import {
  createControlLayerCheckContext,
  createControlLayerSequence,
  describeControlLayerIssue,
  type ControlLayerIssue,
} from './controlLayerChecks';

/** Canvas content a hosted model cannot take, as Generation words it. */
export type HostedCanvasRefusal = Parameters<typeof describeExternalCanvasRefusal>[1];

/** Why the canvas's layers keep generation from running, kept structured until the shell words it. */
export type CanvasInputIssue =
  | ({ kind: 'control_layer' } & ControlLayerIssue)
  | { kind: 'hosted_refusal'; modelName: string; refusal: HostedCanvasRefusal }
  /** Generation words this one; it has no locale key. */
  | { kind: 'layout_prompt'; message: string };

export type ControlLayerVerdict =
  | { status: 'blocked'; issue: CanvasInputIssue }
  /** Judged against no model list it would misjudge every layer: readiness waits, an invocation refuses. */
  | { status: 'awaiting_models'; issue: CanvasInputIssue }
  /** No type and no model: it warns in the panel but contributes nothing. */
  | { status: 'skipped' }
  | { status: 'contributes'; input: Omit<ControlLayerGraphInput, 'imageName'> };

export type RegionalLayerVerdict =
  | { status: 'blocked'; issue: CanvasInputIssue }
  /** Contributes nothing the base can use. */
  | { status: 'skipped' }
  | { status: 'contributes'; input: Omit<RegionalGuidanceInput, 'maskImageName' | 'maskBounds'> };

export interface CanvasInputJudgeParams {
  model: GenerateModelConfig;
  /** Undefined while the model list loads. */
  models: readonly ModelConfig[] | undefined;
  controlModelMemory: ControlModelMemory;
  cfgScale: number;
}

/**
 * Judges each layer generation would composite, as Invoke readiness and the invocation both do. Call `control` in
 * generation order on one judge per invocation: per-generation limits count only the layers that passed before.
 */
export interface CanvasInputJudge {
  control(layer: CanvasControlLayerContract): ControlLayerVerdict;
  regional(layer: CanvasRegionalGuidanceLayerContract): RegionalLayerVerdict;
}

/** Unlike Generation's `toModelIdentifier`, never carries `submodel_type`: these graph inputs have not. */
const toModelReference = (model: { base: string; key: string; name: string; type: string; hash?: unknown }) => ({
  base: model.base,
  key: model.key,
  name: model.name,
  type: model.type,
  ...(typeof model.hash === 'string' ? { hash: model.hash } : {}),
});

/** A region's enabled reference images the base can read, with an image and a model of its base; others drop out. */
const resolveRegionalReferenceImages = (
  region: CanvasRegionalGuidanceLayerContract,
  base: string
): RegionalReferenceImageInput[] => {
  const inputs: RegionalReferenceImageInput[] = [];
  const kind = getRegionalGuidanceSupport(base)?.referenceImages ?? null;
  for (const ref of region.referenceImages) {
    if (!ref.isEnabled) {
      continue;
    }
    const { config } = ref;
    if (config.type === 'ip_adapter' && kind === 'ip_adapter') {
      if (!config.image || !config.model || config.model.base !== base) {
        continue;
      }
      inputs.push({
        beginEndStepPct: config.beginEndStepPct,
        clipVisionModel: config.clipVisionModel,
        id: ref.id,
        imageName: config.image.imageName,
        method: config.method,
        model: toModelReference(config.model),
        type: 'ip_adapter',
        weight: config.weight,
      });
    } else if (config.type === 'flux_redux' && kind === 'flux_redux') {
      if (!config.image || !config.model || config.model.base !== base) {
        continue;
      }
      inputs.push({
        id: ref.id,
        imageName: config.image.imageName,
        model: toModelReference(config.model),
        settings: FLUX_REDUX_INFLUENCE[config.imageInfluence],
        type: 'flux_redux',
      });
    }
  }
  return inputs;
};

export const createCanvasInputJudge = ({
  cfgScale,
  controlModelMemory,
  model,
  models,
}: CanvasInputJudgeParams): CanvasInputJudge => {
  const isHosted = model.type === 'external_image_generator';
  const judgeControl = models
    ? createControlLayerSequence(model, models, createControlLayerCheckContext(model.base, controlModelMemory))
    : null;
  const controlIssue = (layer: CanvasControlLayerContract, code: ControlLayerIssue['code']): CanvasInputIssue => ({
    code,
    kind: 'control_layer',
    layerId: layer.id,
    layerName: layer.name,
  });

  return {
    control: (layer) => {
      if (isHosted) {
        return {
          issue: {
            kind: 'hosted_refusal',
            modelName: model.name,
            refusal: { kind: 'control_layer', layerName: layer.name },
          },
          status: 'blocked',
        };
      }
      if (!judgeControl) {
        return { issue: controlIssue(layer, 'models_unavailable'), status: 'awaiting_models' };
      }
      const { reason, resolution, skip } = judgeControl(layer);
      if (skip) {
        return { status: 'skipped' };
      }
      const resolved = resolution.model;
      if (reason || !resolved) {
        return { issue: controlIssue(layer, reason ?? 'missing_model'), status: 'blocked' };
      }
      const { adapter } = layer;
      return {
        input: {
          beginEndStepPct: adapter.beginEndStepPct,
          controlMode: adapter.controlMode,
          id: layer.id,
          instantxControlMode: resolution.instantxControlMode,
          kind: resolution.kind,
          model: toModelReference(resolved),
          modelCondInChannels: resolved.cond_in_channels,
          modelControlArchitecture: resolved.control_architecture,
          weight: adapter.weight,
        },
        status: 'contributes',
      };
    },
    regional: (layer) => {
      if (isHosted) {
        return {
          issue: {
            kind: 'hosted_refusal',
            modelName: model.name,
            refusal: { kind: 'regional_guidance', layerName: layer.name },
          },
          status: 'blocked',
        };
      }
      const referenceImages = resolveRegionalReferenceImages(layer, model.base);
      const rejection = getRegionalGuidanceRejectionReason({
        cfgScale,
        hasContent: true,
        layerName: layer.name,
        layout: layer.layout,
        mainBase: model.base,
        negativePrompt: layer.negativePrompt,
        positivePrompt: layer.positivePrompt,
        referenceImageCount: referenceImages.length,
      });
      if (rejection) {
        return { status: 'skipped' };
      }
      return {
        input: {
          autoNegative: layer.autoNegative,
          id: layer.id,
          // Typed against Generation's layout roles: a persisted role the graph cannot send fails to compile here.
          ...(layer.layout ? { layout: layer.layout } : {}),
          negativePrompt: layer.negativePrompt,
          positivePrompt: layer.positivePrompt,
          referenceImages,
        },
        status: 'contributes',
      };
    },
  };
};

/**
 * What blocks Invoke before anything is captured, in the order it is listed: raster content a text-only hosted model
 * cannot take, then each compositable layer the judge blocks, in layer order, then a layout built around a blank
 * prompt. Control layers wait for the model list rather than block on its absence.
 */
export const getCanvasInputIssues = (
  params: CanvasInputJudgeParams & {
    /** The contributing layers, in generation order. */
    layers: readonly CanvasLayerContract[];
    /** Whether visible raster content overlaps the frame; measured only for a text-only hosted model. */
    rasterInFrame: boolean;
    /** The compositable regions whose masks reach into the frame, which a layout base places. */
    regionalLayersInFrame: readonly CanvasRegionalGuidanceLayerContract[];
    /** The positive prompt as submitted, template applied. */
    prompt: string;
  }
): CanvasInputIssue[] => {
  const { layers, model, prompt, rasterInFrame, regionalLayersInFrame } = params;
  const judge = createCanvasInputJudge(params);
  const issues: CanvasInputIssue[] = [];

  if (rasterInFrame && isExternalTextOnlyCanvasModel(model)) {
    issues.push({ kind: 'hosted_refusal', modelName: model.name, refusal: { kind: 'raster_content' } });
  }
  for (const layer of layers) {
    const verdict = isCompositableControlLayer(layer)
      ? judge.control(layer)
      : isCompositableRegionalGuidanceLayer(layer)
        ? judge.regional(layer)
        : null;
    if (verdict?.status === 'blocked') {
      issues.push(verdict.issue);
    }
  }
  const layoutPromptReason = getLayoutPromptRejectionReason({
    base: model.base,
    prompt,
    regions: regionalLayersInFrame,
  });
  if (layoutPromptReason) {
    issues.push({ kind: 'layout_prompt', message: layoutPromptReason });
  }
  return issues;
};

type Translate = (key: string, options?: Record<string, unknown>) => string;

/** The one wording of a canvas input issue, for the Invoke tooltip and the failed-invoke notice alike. */
export const describeCanvasInputIssue = (t: Translate, issue: CanvasInputIssue): string => {
  switch (issue.kind) {
    case 'control_layer':
      return describeControlLayerIssue(t, issue);
    case 'hosted_refusal':
      return describeExternalCanvasRefusal({ name: issue.modelName }, issue.refusal);
    case 'layout_prompt':
      return issue.message;
  }
};
