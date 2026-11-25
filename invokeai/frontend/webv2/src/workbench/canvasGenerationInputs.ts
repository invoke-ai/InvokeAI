import type { GenerateModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type {
  CanvasControlLayerContract,
  CanvasInpaintMaskLayerContract,
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

export type HostedCanvasRefusal = Parameters<typeof describeExternalCanvasRefusal>[1];

export type CanvasInputIssue =
  | ({ kind: 'control_layer' } & ControlLayerIssue)
  | { kind: 'hosted_refusal'; modelName: string; refusal: HostedCanvasRefusal }
  | { kind: 'layout_prompt'; message: string };

export type ControlLayerVerdict =
  | { status: 'blocked'; issue: CanvasInputIssue }
  /** Readiness waits for the model list; an invocation refuses. */
  | { status: 'awaiting_models'; issue: CanvasInputIssue }
  | { status: 'skipped' }
  | { status: 'contributes'; input: Omit<ControlLayerGraphInput, 'imageName'> };

export type RegionalLayerVerdict =
  | { status: 'blocked'; issue: CanvasInputIssue }
  | { status: 'skipped' }
  | { status: 'contributes'; input: Omit<RegionalGuidanceInput, 'maskImageName' | 'maskBounds'> };

export interface CanvasInputJudgeParams {
  model: GenerateModelConfig;
  models: readonly ModelConfig[] | undefined;
  controlModelMemory: ControlModelMemory;
  cfgScale: number;
}

/** Call `control` in generation order on one judge per invocation: per-generation limits count passed layers. */
export interface CanvasInputJudge {
  control(layer: CanvasControlLayerContract): ControlLayerVerdict;
  regional(layer: CanvasRegionalGuidanceLayerContract): RegionalLayerVerdict;
  /** A mask's guidance as a region over its coverage; the caller asks only while the mask inpaints. */
  maskGuidance(layer: CanvasInpaintMaskLayerContract): RegionalLayerVerdict;
}

/** Unlike Generation's `toModelIdentifier`: these graph inputs carry no `submodel_type`. */
const toModelReference = (model: { base: string; key: string; name: string; type: string; hash?: unknown }) => ({
  base: model.base,
  key: model.key,
  name: model.name,
  type: model.type,
  ...(typeof model.hash === 'string' ? { hash: model.hash } : {}),
});

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

  const judgeRegion = (
    layerName: string,
    input: Extract<RegionalLayerVerdict, { status: 'contributes' }>['input']
  ): RegionalLayerVerdict => {
    const rejection = getRegionalGuidanceRejectionReason({
      cfgScale,
      hasContent: true,
      layerName,
      layout: input.layout,
      mainBase: model.base,
      negativePrompt: input.negativePrompt,
      positivePrompt: input.positivePrompt,
      referenceImageCount: input.referenceImages.length,
    });
    return rejection ? { status: 'skipped' } : { input, status: 'contributes' };
  };

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
      return judgeRegion(layer.name, {
        autoNegative: layer.autoNegative,
        id: layer.id,
        // Typed against Generation's layout roles: a persisted role the graph cannot send fails to compile here.
        ...(layer.layout ? { layout: layer.layout } : {}),
        negativePrompt: layer.negativePrompt,
        positivePrompt: layer.positivePrompt,
        referenceImages: resolveRegionalReferenceImages(layer, model.base),
      });
    },
    maskGuidance: (layer) => {
      const { guidance } = layer;
      // Hosted inpainting is refused for the mask itself, so its guidance adds no refusal of its own.
      if (isHosted || !guidance?.isEnabled) {
        return { status: 'skipped' };
      }
      return judgeRegion(layer.name, {
        autoNegative: guidance.autoNegative,
        id: layer.id,
        negativePrompt: guidance.negativePrompt,
        positivePrompt: guidance.positivePrompt,
        referenceImages: [],
      });
    },
  };
};

/** Control layers wait for the model list rather than block on its absence. */
export const getCanvasInputIssues = (
  params: CanvasInputJudgeParams & {
    layers: readonly CanvasLayerContract[];
    /** Measured only for a text-only hosted model. */
    rasterInFrame: boolean;
    regionalLayersInFrame: readonly CanvasRegionalGuidanceLayerContract[];
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
