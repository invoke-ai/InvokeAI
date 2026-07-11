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

/** Normalizes the new adapter shape without reinterpreting any older persisted kind. */
export const normalizeControlAdapter = (value: unknown): unknown => {
  if (!isRecord(value) || value.kind !== 'z_image_control') {
    return value;
  }
  const defaults = CONTROL_ADAPTER_DEFAULTS.z_image_control;
  const range = value.beginEndStepPct;
  return {
    beginEndStepPct:
      Array.isArray(range) && range.length === 2 && range.every((item) => typeof item === 'number')
        ? [range[0], range[1]]
        : [...defaults.beginEndStepPct],
    controlMode: null,
    kind: 'z_image_control',
    model: typeof value.model === 'string' ? value.model : null,
    weight: typeof value.weight === 'number' ? value.weight : defaults.weight,
  } satisfies CanvasControlAdapterContract;
};
