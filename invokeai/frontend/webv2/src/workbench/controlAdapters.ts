import type { CanvasControlAdapterContract } from './types';

export const CONTROL_ADAPTER_DEFAULTS: Readonly<
  Record<CanvasControlAdapterContract['kind'], CanvasControlAdapterContract>
> = {
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

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const isControlAdapterKind = (value: unknown): value is CanvasControlAdapterContract['kind'] =>
  typeof value === 'string' && Object.hasOwn(CONTROL_ADAPTER_DEFAULTS, value);

export const areControlAdapterValuesValid = (
  kind: CanvasControlAdapterContract['kind'],
  weight: unknown,
  beginEndStepPct: unknown
): beginEndStepPct is [number, number] => {
  const minWeight = kind === 'z_image_control' ? 0 : -1;
  if (typeof weight !== 'number' || !Number.isFinite(weight) || weight < minWeight || weight > 2) {
    return false;
  }
  if (!Array.isArray(beginEndStepPct) || beginEndStepPct.length !== 2) {
    return false;
  }
  const [begin, end] = beginEndStepPct;
  return (
    typeof begin === 'number' &&
    Number.isFinite(begin) &&
    begin >= 0 &&
    begin <= 1 &&
    typeof end === 'number' &&
    Number.isFinite(end) &&
    end >= 0 &&
    end <= 1 &&
    begin < end
  );
};

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
