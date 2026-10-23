/**
 * What every row of the control layer Properties panel shows, derived from the layer, its model resolution, the
 * installed models and detection status. Pure: texts are locale keys under `widgets.layers.control.` with values and
 * nested keys (`refs`) the view translates, so tests read states without a locale.
 */

import type { ControlAdapterKind } from '@features/generation/graph';
import type { CanvasControlLayerContract, ControlType } from '@workbench/canvas-engine/api';
import type { ControlDetectionStatus } from '@workbench/canvas-operations/api';
import type { ControlImageState } from '@workbench/canvasLayerContent';
import type { ControlLayerIssueCode, ControlMapState } from '@workbench/controlLayerChecks';
import type { ControlModelResolution, ControlTypeAvailability } from '@workbench/controlModelResolution';

import { CONTROL_TYPES } from '@workbench/canvas-engine/api';
import { CONTROL_TYPE_SPECS, classifyControlModel, getDetectorFamily } from '@workbench/controlTypes';

/** A text the view translates: `key` and `refs` values are keys relative to `widgets.layers.control.`. */
export interface PanelText {
  key: string;
  values?: Record<string, string | number>;
  /** Placeholders whose value is itself a locale key under `widgets.layers.control.`. */
  refs?: Record<string, string>;
  /** Placeholders the view joins as a disjunctive list ("A, B or C"). */
  lists?: Record<string, readonly string[]>;
}

export type PanelTone = 'default' | 'muted' | 'warning' | 'error';

/** Detection's "None: use the image as it is" entry. */
export const DETECTION_NONE = 'none';

export interface DetectionOption {
  value: string;
  label: PanelText;
  disabled: boolean;
  /** Why a detector cannot be chosen now. */
  hint: PanelText | null;
}

export type ControlModelAction = 'reset' | 'find-one' | 'install-starters';

export type ControlPriority = 'more_prompt' | 'balanced' | 'more_control';

export interface ControlPanelModel {
  image: { state: ControlImageState; imageName: string | null };
  type: {
    value: ControlType | null;
    /** An image is there but nothing tells the layer what it controls, so it contributes nothing. */
    invalid: boolean;
    hints: Readonly<Record<ControlType, { text: PanelText; tone: PanelTone }>>;
  };
  detection: {
    /** No type yet: the detector follows from it. */
    disabled: boolean;
    value: string | null;
    valueText: PanelText;
    tone: PanelTone;
    options: readonly DetectionOption[];
    /** The detector queued or running. */
    busy: { detector: string | null } | null;
    failed: { code: string } | null;
    /** `retry` after a failure; `detect` when the map is out of date and nothing is scheduled. */
    action: 'retry' | 'detect' | null;
    /** The detector whose settings the inline region edits; null with the reason it cannot. */
    settings: { detector: string; values: Record<string, unknown> } | null;
    settingsUnavailable: PanelText | null;
  };
  model: {
    /** The resolved model's name, or a text standing in for it. */
    name: string | null;
    valueText: PanelText | null;
    tone: PanelTone;
    caption: PanelText | null;
    captionTone: PanelTone;
    action: ControlModelAction | null;
  };
  adapter: {
    kind: ControlAdapterKind;
    priority: { disabled: boolean; value: ControlPriority | null };
    stepsReadout: PanelText;
  };
}

export interface ControlPanelInput {
  layer: CanvasControlLayerContract;
  imageState: ControlImageState;
  resolution: ControlModelResolution;
  map: ControlMapState;
  /** The reason to show: Invoke's judgement for a contributing layer, the ambient one otherwise. */
  reason: ControlLayerIssueCode | null;
  availability: readonly ControlTypeAvailability[];
  status: ControlDetectionStatus;
  /** The main model's base and its display name; null without a main model. */
  base: { id: string; label: string } | null;
  /** Installed models of the base held back from the list (LLLite inpainting or unidentified adapters). */
  hiddenModelReasons?: readonly string[];
}

/** Reasons the Type and Model rows already say in their own words. */
const RESOLUTION_REASONS: ReadonlySet<ControlLayerIssueCode> = new Set([
  'missing_model',
  'missing_type',
  'choose_model',
  'no_model_for_type',
  'no_control_models',
]);
/** Reasons shown elsewhere: detection on its row, capabilities in their own block. */
const ELSEWHERE_REASONS: ReadonlySet<ControlLayerIssueCode> = new Set([
  'detection_pending',
  'capabilities_unavailable',
]);

const typeHint = (entry: ControlTypeAvailability | undefined): { text: PanelText; tone: PanelTone } => {
  switch (entry?.state) {
    case 'ready':
    case 'ask':
      return { text: { key: 'typeHint.models', values: { count: entry.count } }, tone: 'muted' };
    case 'union':
      return { text: { key: 'typeHint.union' }, tone: 'muted' };
    default:
      return { text: { key: 'typeHint.notInstalled' }, tone: 'warning' };
  }
};

const deriveType = (input: ControlPanelInput): ControlPanelModel['type'] => {
  const { availability, imageState, layer, resolution } = input;
  const byType = new Map(availability.map((entry) => [entry.type, entry]));
  const hints = Object.fromEntries(CONTROL_TYPES.map((type) => [type, typeHint(byType.get(type))])) as Record<
    ControlType,
    { text: PanelText; tone: PanelTone }
  >;
  return {
    hints,
    // A legacy layer that names a working model still generates; only a layer with nothing to run is flagged.
    invalid: !layer.controlType && imageState !== 'empty' && resolution.unresolved !== null,
    value: layer.controlType ?? null,
  };
};

const filterLabel = (detector: string): PanelText => ({ key: `filters.${detector}` });

const deriveDetection = (input: ControlPanelInput): ControlPanelModel['detection'] => {
  const { imageState, layer, map, status } = input;
  const type = layer.controlType;
  const busy =
    status.status === 'processing'
      ? { detector: status.detector }
      : status.status === 'queued'
        ? { detector: layer.detection?.type ?? null }
        : null;
  const failed = status.status === 'error' ? { code: status.error.code } : null;
  if (!type) {
    return {
      action: null,
      busy: null,
      disabled: true,
      failed: null,
      options: [],
      settings: null,
      settingsUnavailable: { key: 'detectionSetByType' },
      tone: 'muted',
      value: null,
      valueText: { key: 'detectionSetByType' },
    };
  }
  const noneLabel: PanelText = { key: `detectionNone.${type}` };
  // Detection needs pixels it can run on: a map without its original (I5) has none until the image is replaced.
  const detectorHint: PanelText | null =
    imageState === 'empty' ? { key: 'detectionNeedsImage' } : map === 'no-input' ? { key: 'detectionNoInput' } : null;
  const options: DetectionOption[] = [
    ...CONTROL_TYPE_SPECS[type].detectors.map((detector) => ({
      disabled: detectorHint !== null,
      hint: detectorHint,
      label: filterLabel(detector.type),
      value: detector.type,
    })),
    { disabled: false, hint: null, label: noneLabel, value: DETECTION_NONE },
  ];
  const detection = layer.detection ?? null;
  const value = detection ? detection.type : DETECTION_NONE;
  const action = failed ? 'retry' : status.status === 'idle' && map === 'stale' ? 'detect' : null;
  const settings = detection && detectorHint === null ? { detector: detection.type, values: detection.settings } : null;
  return {
    action,
    busy,
    disabled: false,
    failed,
    options,
    settings,
    // A detector with nothing to run on says why; None has no settings at all.
    settingsUnavailable: settings
      ? null
      : detection && detectorHint
        ? detectorHint
        : { key: 'detectionNoSettings', refs: { detector: detection ? `filters.${detection.type}` : noneLabel.key } },
    tone: failed ? 'error' : 'default',
    value,
    valueText: detection ? filterLabel(detection.type) : noneLabel,
  };
};

/** The words for a resolved model that was not made for what the detector produces. */
const mismatchCaption = (input: ControlPanelInput, kindRef: string): PanelText | null => {
  const { layer, resolution } = input;
  if (!resolution.model || resolution.familyMatch || !layer.controlType) {
    return null;
  }
  const modelClass = classifyControlModel(resolution.model);
  if (modelClass.union) {
    return { key: 'modelCaption.unionUncovered', refs: { kind: kindRef, type: `typeNouns.${layer.controlType}` } };
  }
  if (!modelClass.family) {
    return { key: 'modelCaption.unknownPurpose', refs: { kind: kindRef, type: `typeNouns.${layer.controlType}` } };
  }
  const family = getDetectorFamily(layer.controlType, layer.detection?.type ?? null);
  return {
    key: 'modelCaption.mismatch',
    refs: { family: `families.${family}`, kind: kindRef, madeFor: `families.${modelClass.family}` },
  };
};

const deriveModel = (input: ControlPanelInput): ControlPanelModel['model'] => {
  const { base, hiddenModelReasons = [], imageState, layer, reason, resolution } = input;
  const type = layer.controlType;
  const kindRef = `kinds.${resolution.kind}`;
  const shownReason = reason && !RESOLUTION_REASONS.has(reason) && !ELSEWHERE_REASONS.has(reason) ? reason : null;
  // Problems with the model itself (a base mismatch, a per-generation limit) outrank the explanation of the choice.
  const reasonCaption: PanelText | null = shownReason
    ? { key: `validation.${shownReason}`, refs: { kind: kindRef } }
    : null;
  const reasonTone: PanelTone = shownReason === 'models_unavailable' ? 'muted' : 'warning';

  if (!type) {
    if (resolution.model && resolution.unresolved === null) {
      return {
        action: null,
        caption:
          shownReason === 'switch_adapter_kind'
            ? { key: 'modelCaption.chooseType' }
            : (reasonCaption ?? { key: kindRef }),
        captionTone: shownReason ? reasonTone : 'muted',
        name: resolution.model.name,
        tone: 'default',
        valueText: null,
      };
    }
    // No type and no working model: the type is what to fix, so the model waits for it.
    const attention = imageState !== 'empty' || resolution.unresolved === 'missing_model';
    return {
      action: null,
      caption: { key: attention ? 'modelCaption.chooseType' : 'typeMissing' },
      captionTone: attention ? 'warning' : 'muted',
      name: null,
      tone: 'muted',
      valueText: { key: 'modelValue.setByType' },
    };
  }

  const typeRef = `typeNouns.${type}`;
  const baseValues = { base: base?.label ?? '' };
  const withReason = (model: ControlPanelModel['model']): ControlPanelModel['model'] =>
    reasonCaption ? { ...model, caption: reasonCaption, captionTone: reasonTone } : model;
  const mismatch = mismatchCaption(input, kindRef);

  switch (resolution.how) {
    case 'explicit':
    case 'remembered':
      return withReason({
        action: 'reset',
        caption: mismatch ?? { key: 'modelCaption.pick', refs: { kind: kindRef, type: typeRef } },
        captionTone: mismatch ? 'warning' : 'muted',
        name: resolution.model?.name ?? null,
        tone: 'default',
        valueText: null,
      });
    case 'only':
      return withReason({
        action: null,
        caption: mismatch ?? { key: 'modelCaption.only', refs: { kind: kindRef, type: typeRef } },
        captionTone: mismatch ? 'warning' : 'muted',
        name: resolution.model?.name ?? null,
        tone: 'default',
        valueText: null,
      });
    case 'union':
      return withReason({
        action: null,
        caption: { key: 'modelCaption.union', refs: { kind: kindRef, type: typeRef } },
        captionTone: 'muted',
        name: resolution.model?.name ?? null,
        tone: 'default',
        valueText: null,
      });
    case 'ask':
      return withReason({
        action: null,
        caption: {
          key: 'modelCaption.ambiguous',
          lists: { models: resolution.candidates.map((model) => model.name) },
          values: { count: resolution.candidates.length },
        },
        captionTone: 'muted',
        name: null,
        tone: 'warning',
        valueText: { key: 'modelValue.choose', refs: { type: typeRef } },
      });
    default: {
      const noModels = resolution.unresolved === 'no_control_models';
      const hidden = hiddenModelReasons[0];
      return withReason({
        action: hidden ? null : noModels ? 'install-starters' : 'find-one',
        caption: hidden
          ? { key: `hiddenModels.${hidden}` }
          : base
            ? {
                key: noModels ? 'modelCaption.noModels' : 'modelCaption.noneForBase',
                refs: { type: typeRef },
                values: baseValues,
              }
            : { key: 'modelCaption.noModelsAnyBase' },
        captionTone: 'warning',
        name: null,
        tone: 'warning',
        valueText:
          noModels || !base ? { key: 'modelValue.noModels' } : { key: 'modelValue.noneForBase', values: baseValues },
      });
    }
  }
};

const percent = (value: number): number => Math.round(value * 100);

const deriveAdapter = (input: ControlPanelInput): ControlPanelModel['adapter'] => {
  const { layer, resolution } = input;
  const { beginEndStepPct, controlMode } = layer.adapter;
  const isControlNet = resolution.kind === 'controlnet';
  return {
    kind: resolution.kind,
    priority: {
      disabled: !isControlNet,
      // Legacy `unbalanced` has no segment and shows none checked; ControlNet without a stored mode runs balanced.
      value: controlMode === 'unbalanced' ? null : (controlMode ?? (isControlNet ? 'balanced' : null)),
    },
    stepsReadout: {
      key: 'stepsReadout',
      values: { begin: percent(beginEndStepPct[0]), end: percent(beginEndStepPct[1]) },
    },
  };
};

export const deriveControlPanel = (input: ControlPanelInput): ControlPanelModel => {
  const pixels = input.layer.original ?? input.layer.source;
  return {
    adapter: deriveAdapter(input),
    detection: deriveDetection(input),
    image: { imageName: pixels.type === 'image' ? pixels.image.imageName : null, state: input.imageState },
    model: deriveModel(input),
    type: deriveType(input),
  };
};

type Translate = (key: string, options?: Record<string, unknown>) => string;

const KEY_PREFIX = 'widgets.layers.control.';

/** Translates a panel text; lists join as "A, B or C" in `locale`. */
export const formatPanelText = (t: Translate, text: PanelText, locale: string): string => {
  const options: Record<string, unknown> = { ...text.values };
  for (const [name, key] of Object.entries(text.refs ?? {})) {
    options[name] = t(KEY_PREFIX + key);
  }
  for (const [name, items] of Object.entries(text.lists ?? {})) {
    options[name] = new Intl.ListFormat(locale, { type: 'disjunction' }).format(items);
  }
  return t(KEY_PREFIX + text.key, options);
};
