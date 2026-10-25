import type { ControlAdapterKind } from '@features/generation/graph';
import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract, ControlType } from '@workbench/canvas-engine/api';

import { getControlModelUnusableReason } from '@features/generation/graph';
import { CONTROL_TYPES } from '@workbench/canvas-engine/api';

import { deriveControlKind, getDefaultControlAdapterKind } from './controlAdapters';
import { CONTROL_TYPE_SPECS, classifyControlModel, getDetectorFamily, type ControlFamily } from './controlTypes';

/** The user's last model pick per main-model base and control type: `${base}:${controlType}` → model key. */
export type ControlModelMemory = Readonly<Record<string, string>>;

export const controlModelMemoryKey = (base: string, type: ControlType): string => `${base}:${type}`;

/**
 * Why no model drives a layer; validation reports it in place of `missing_model`. `missing_type`: no type and no
 * model, so the layer contributes nothing. `missing_model`: an untyped layer names a model that is not installed.
 */
export type ControlModelUnresolvedReason =
  | 'missing_type'
  | 'missing_model'
  | 'choose_model'
  | 'no_model_for_type'
  | 'no_control_models';

export interface ControlModelResolution {
  /**
   * `explicit`: the layer's own pick. `remembered`: the last pick for this base and type, still among `candidates`;
   * it applies to every layer of that base and type without an explicit pick. `only`: the one model made for the
   * type. `union`: the one union model covering it. `ask`: several fit. `missing`: none fit. `none`: the layer has
   * no type, so its stored adapter applies as it is.
   */
  how: 'explicit' | 'remembered' | 'only' | 'union' | 'ask' | 'missing' | 'none';
  model: ModelConfig | null;
  /** The kind the layer runs as: the model's own, else the base's default (the stored kind for untyped layers). */
  kind: ControlAdapterKind;
  /** Models that fit the type: those made for it first, then covering unions. */
  candidates: readonly ModelConfig[];
  /** Every model the layer could run on this base, whatever it was made for, typed or not. */
  usable: readonly ModelConfig[];
  /**
   * Whether the model was made for (or, as a union, covers) the family the detector produces; false when it only
   * shares the type's purpose (a normal-map model on a depth layer), and for untyped or unresolved layers.
   */
  familyMatch: boolean;
  /** FLUX InstantX Union mode for this type's family; null for every other model. */
  instantxControlMode: number | null;
  unresolved: ControlModelUnresolvedReason | null;
}

type SupportsKind = (kind: ControlAdapterKind) => boolean;

interface CandidateSets {
  usable: ModelConfig[];
  dedicated: ModelConfig[];
  unions: ModelConfig[];
}

/** A model a layer on `base` can run: same base, a control kind the base supports, and nothing about it refused. */
const usableKind = (model: ModelConfig, base: string | null, supportsKind: SupportsKind): ControlAdapterKind | null => {
  if (base !== null && model.base !== base) {
    return null;
  }
  const kind = deriveControlKind(model);
  return kind && supportsKind(kind) && getControlModelUnusableReason(model, kind) === null ? kind : null;
};

const usableModels = (base: string | null, models: readonly ModelConfig[], supportsKind: SupportsKind) =>
  models.filter((model) => usableKind(model, base, supportsKind) !== null);

const collectCandidates = (
  type: ControlType,
  family: ControlFamily,
  base: string | null,
  models: readonly ModelConfig[],
  supportsKind: SupportsKind
): CandidateSets => {
  const usable = usableModels(base, models, supportsKind);
  const classes = new Map(usable.map((model) => [model, classifyControlModel(model)]));
  const byFamily = usable.filter((model) => classes.get(model)!.family === family);
  return {
    dedicated: byFamily.length > 0 ? byFamily : usable.filter((model) => classes.get(model)!.purpose === type),
    unions: usable.filter((model) => {
      const modelClass = classes.get(model)!;
      return modelClass.union && modelClass.covers.includes(family);
    }),
    usable,
  };
};

/**
 * Which model drives a control layer: an explicit pick for this base, the remembered pick for (base, type), the
 * only model made for the type, a choice between several, the one covering union model, or none. A pick made for
 * another base is ignored, not cleared, so switching back restores it. An untyped layer runs its stored adapter as
 * it is; its `usable` still lists every model it could run on the base, for a direct pick.
 */
export const resolveControlModel = (input: {
  layer: Pick<CanvasControlLayerContract, 'controlType' | 'detection' | 'adapter'>;
  base: string | null;
  models: readonly ModelConfig[];
  remembered: ControlModelMemory;
  supportsKind: SupportsKind;
}): ControlModelResolution => {
  const { base, layer, models, remembered, supportsKind } = input;
  const picked = layer.adapter.model ? (models.find((model) => model.key === layer.adapter.model) ?? null) : null;
  const type = layer.controlType;

  if (!type) {
    return {
      candidates: [],
      familyMatch: false,
      how: 'none',
      instantxControlMode: null,
      kind: layer.adapter.kind,
      model: picked,
      unresolved: layer.adapter.model ? (picked ? null : 'missing_model') : 'missing_type',
      usable: usableModels(base, models, supportsKind),
    };
  }

  const family = getDetectorFamily(type, layer.detection?.type ?? null);
  const { dedicated, unions, usable } = collectCandidates(type, family, base, models, supportsKind);
  const candidates = [...dedicated, ...unions];
  const settle = (
    how: ControlModelResolution['how'],
    model: ModelConfig | null,
    unresolved: ControlModelUnresolvedReason | null = null
  ): ControlModelResolution => {
    const modelClass = model ? classifyControlModel(model) : null;
    return {
      candidates,
      familyMatch:
        modelClass !== null &&
        (modelClass.family === family || (modelClass.union && modelClass.covers.includes(family))),
      how,
      instantxControlMode:
        base === 'flux' && modelClass?.union ? (CONTROL_TYPE_SPECS[type].instantxMode[family] ?? null) : null,
      kind: (model ? deriveControlKind(model) : null) ?? getDefaultControlAdapterKind(base),
      model,
      unresolved,
      usable,
    };
  };

  if (picked && usableKind(picked, base, supportsKind)) {
    return settle('explicit', picked);
  }
  const rememberedKey = base === null ? undefined : remembered[controlModelMemoryKey(base, type)];
  const rememberedModel = rememberedKey ? candidates.find((model) => model.key === rememberedKey) : undefined;
  if (rememberedModel) {
    return settle('remembered', rememberedModel);
  }
  if (dedicated.length === 1) {
    return settle('only', dedicated[0]!);
  }
  if (dedicated.length > 1) {
    return settle('ask', null, 'choose_model');
  }
  if (unions.length === 1) {
    return settle('union', unions[0]!);
  }
  if (unions.length > 1) {
    return settle('ask', null, 'choose_model');
  }
  return settle('missing', null, usable.length === 0 ? 'no_control_models' : 'no_model_for_type');
};

/**
 * Whether picking `model` should become the remembered pick for the layer's base and type. Resolution honours a
 * remembered pick only among `candidates`, and it then drives every unpicked layer of that base and type, so a model
 * made for another family, or one nothing identifies, stays this layer's own pick.
 */
export const isRememberableControlPick = (resolution: ControlModelResolution, model: ModelConfig): boolean =>
  resolution.candidates.some((candidate) => candidate.key === model.key);

/** How a control type would resolve on a fresh layer, for the Type menu's hints. */
export interface ControlTypeAvailability {
  type: ControlType;
  /** `ready`: one model made for it. `ask`: several. `union`: only union models cover it. `missing`: nothing. */
  state: 'ready' | 'ask' | 'union' | 'missing';
  /** Models behind the state: those made for the type, or the covering unions. */
  count: number;
}

export const getTypeAvailability = (
  base: string | null,
  models: readonly ModelConfig[],
  supportsKind: SupportsKind
): readonly ControlTypeAvailability[] =>
  CONTROL_TYPES.map((type) => {
    const { dedicated, unions } = collectCandidates(
      type,
      CONTROL_TYPE_SPECS[type].primaryFamily,
      base,
      models,
      supportsKind
    );
    if (dedicated.length > 0) {
      return { count: dedicated.length, state: dedicated.length === 1 ? 'ready' : 'ask', type };
    }
    return unions.length > 0 ? { count: unions.length, state: 'union', type } : { count: 0, state: 'missing', type };
  });
