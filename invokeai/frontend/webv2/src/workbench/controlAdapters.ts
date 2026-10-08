import type { CanvasControlAdapterContract } from '@workbench/canvas-engine/api';

import { areControlAdapterValuesValid } from '@features/generation/graph';

export const CONTROL_ADAPTER_DEFAULTS: Readonly<
  Record<CanvasControlAdapterContract['kind'], CanvasControlAdapterContract>
> = {
  anima_lllite: {
    beginEndStepPct: [0, 1],
    controlMode: null,
    kind: 'anima_lllite',
    model: null,
    weight: 1,
  },
  control_lora: {
    beginEndStepPct: [0, 1],
    controlMode: null,
    kind: 'control_lora',
    model: null,
    weight: 0.75,
  },
  controlnet: {
    beginEndStepPct: [0, 0.75],
    controlMode: 'balanced',
    kind: 'controlnet',
    model: null,
    weight: 0.75,
  },
  t2i_adapter: {
    beginEndStepPct: [0, 1],
    controlMode: null,
    kind: 't2i_adapter',
    model: null,
    weight: 1,
  },
  z_image_control: {
    beginEndStepPct: [0, 1],
    controlMode: null,
    kind: 'z_image_control',
    model: null,
    weight: 0.75,
  },
};

/** Kinds that exist for exactly one main-model base. Every other kind is offered wherever capabilities allow it. */
export const CONTROL_KIND_BASE: Readonly<Partial<Record<CanvasControlAdapterContract['kind'], string>>> = {
  anima_lllite: 'anima',
  z_image_control: 'z-image',
};

/**
 * The adapter kind a new control layer starts with for a main-model base: a base's own kind, otherwise ControlNet. A
 * layer keeps its kind when the main model changes; validation reports a mismatch.
 */
export const getDefaultControlAdapterKind = (base: string | null | undefined): CanvasControlAdapterContract['kind'] =>
  (Object.keys(CONTROL_KIND_BASE) as CanvasControlAdapterContract['kind'][]).find(
    (kind) => CONTROL_KIND_BASE[kind] === base
  ) ?? 'controlnet';

/** A fresh copy of the default adapter for `base`, carrying `model`. */
export const createDefaultControlAdapter = (
  base: string | null | undefined,
  model: string | null = null
): CanvasControlAdapterContract => {
  const defaults = CONTROL_ADAPTER_DEFAULTS[getDefaultControlAdapterKind(base)];
  return { ...defaults, beginEndStepPct: [...defaults.beginEndStepPct], model };
};

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const isControlAdapterKind = (value: unknown): value is CanvasControlAdapterContract['kind'] =>
  typeof value === 'string' && Object.hasOwn(CONTROL_ADAPTER_DEFAULTS, value);

/** Repairs persisted numeric fields while preserving valid values and adapter kinds. */
export const normalizeControlAdapter = (value: unknown): unknown => {
  if (!isRecord(value) || !isControlAdapterKind(value.kind)) {
    return value;
  }
  const defaults = CONTROL_ADAPTER_DEFAULTS[value.kind];
  const minWeight = value.kind === 'z_image_control' ? 0 : -1;
  const weight =
    typeof value.weight === 'number' && Number.isFinite(value.weight) && value.weight >= minWeight && value.weight <= 2
      ? value.weight
      : defaults.weight;
  const beginEndStepPct: [number, number] = areControlAdapterValuesValid(value.kind, weight, value.beginEndStepPct)
    ? [value.beginEndStepPct[0], value.beginEndStepPct[1]]
    : [defaults.beginEndStepPct[0], defaults.beginEndStepPct[1]];

  if (value.kind !== 'z_image_control') {
    return { ...defaults, ...value, beginEndStepPct, weight } satisfies CanvasControlAdapterContract;
  }
  return {
    beginEndStepPct,
    controlMode: null,
    kind: 'z_image_control',
    model: typeof value.model === 'string' ? value.model : null,
    weight,
  } satisfies CanvasControlAdapterContract;
};
