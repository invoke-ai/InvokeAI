import type { CanvasControlAdapterContract } from '@workbench/canvas-engine/api';

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
  // The FLUX.2 [dev] Fun-ControlNet-Union's recommended 0.65–0.80 context scale; the node's own default.
  flux2_control: {
    beginEndStepPct: [0, 1],
    controlMode: null,
    kind: 'flux2_control',
    model: null,
    weight: 0.75,
  },
  // The Fun-ControlNet-Union's recommended 0.70–0.95 context scale, centred.
  qwen_image_control: {
    beginEndStepPct: [0, 1],
    controlMode: null,
    kind: 'qwen_image_control',
    model: null,
    weight: 0.85,
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
  flux2_control: 'flux2',
  qwen_image_control: 'qwen-image',
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

/**
 * The adapter kind a control model runs as on its own base: T2I-Adapter and Control LoRA by type, a `controlnet`
 * model as its base's own kind (Z-Image, Qwen-Image, FLUX.2, Anima LLLite) or plain ControlNet. Null for other types.
 */
export const deriveControlKind = (model: {
  type: string;
  base: string;
}): CanvasControlAdapterContract['kind'] | null => {
  if (model.type === 't2i_adapter' || model.type === 'control_lora') {
    return model.type;
  }
  return model.type === 'controlnet' ? getDefaultControlAdapterKind(model.base) : null;
};

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

const CONTROL_MODES: readonly unknown[] = ['balanced', 'more_prompt', 'more_control', 'unbalanced'];

/**
 * Whether a stored weight can be kept: finite and within -1..2, the widest range any kind accepts. A stored adapter's
 * kind is advisory (the layer runs as the kind its model resolves to, which the panel, readiness validation and the
 * graph all use), so persistence must not judge the weight by it: a Fun kind's 0 floor would refuse or reset a
 * negative weight a layer created under that base may now validly run with.
 */
export const isStorableControlWeight = (weight: unknown): weight is number =>
  typeof weight === 'number' && Number.isFinite(weight) && weight >= -1 && weight <= 2;

/** Whether a stored begin/end step range can be kept: finite, within 0..1 and ordered; the same for every kind. */
export const isStorableControlStepRange = (range: unknown): range is [number, number] =>
  Array.isArray(range) &&
  range.length === 2 &&
  typeof range[0] === 'number' &&
  typeof range[1] === 'number' &&
  Number.isFinite(range[0]) &&
  Number.isFinite(range[1]) &&
  range[0] >= 0 &&
  range[1] <= 1 &&
  range[0] < range[1];

/**
 * Maps a persisted adapter to the contract, keeping every stored value persistence can hold and repairing the rest
 * with the stored kind's defaults. No kind-specific reset applies (see {@link isStorableControlWeight}): the stored
 * kind and model are kept as they are, and the resolved kind governs what runs.
 */
export const normalizeControlAdapter = (value: unknown): unknown => {
  if (!isRecord(value) || !isControlAdapterKind(value.kind)) {
    return value;
  }
  const defaults = CONTROL_ADAPTER_DEFAULTS[value.kind];
  return {
    beginEndStepPct: isStorableControlStepRange(value.beginEndStepPct)
      ? [value.beginEndStepPct[0], value.beginEndStepPct[1]]
      : [defaults.beginEndStepPct[0], defaults.beginEndStepPct[1]],
    controlMode:
      value.controlMode === null || CONTROL_MODES.includes(value.controlMode)
        ? (value.controlMode as CanvasControlAdapterContract['controlMode'])
        : defaults.controlMode,
    kind: value.kind,
    model: typeof value.model === 'string' ? value.model : null,
    weight: isStorableControlWeight(value.weight) ? value.weight : defaults.weight,
  } satisfies CanvasControlAdapterContract;
};
