import {
  getArchitectureFeatures,
  hasArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';

/** Every adapter kind a control layer can carry, in the order pickers list them. */
export const CONTROL_ADAPTER_KINDS = [
  'controlnet',
  't2i_adapter',
  'control_lora',
  'z_image_control',
  'anima_lllite',
] as const;
export type ControlAdapterKind = (typeof CONTROL_ADAPTER_KINDS)[number];

/** Every reason a control layer can be rejected; each has a `widgets.layers.control.validation` locale string. */
export const CONTROL_VALIDATION_REASONS = [
  'capabilities_unavailable',
  'missing_model',
  'switch_adapter_kind',
  'unsupported_adapter',
  'incompatible_base',
  'incompatible_adapter',
  'lllite_inpaint_adapter',
  'lllite_channels_unknown',
  'invalid_adapter_values',
  'control_lora_limit',
  'z_image_control_limit',
  'duplicate_lllite_model',
  'flux_fill_control_lora',
] as const;
export type ControlValidationReason = (typeof CONTROL_VALIDATION_REASONS)[number];

/** What validation needs from a resolved adapter model; a backend model config satisfies it directly. */
export interface ControlAdapterModelFacts {
  base: string;
  type: string;
  /** Anima ControlNet-LLLite only: 3 = control image, 4 = inpainting; null for installs that predate the field. */
  cond_in_channels?: number | null;
}

/** Z-Image control and Anima ControlNet-LLLite models install with the `controlnet` model type. */
const getControlAdapterModelType = (kind: ControlAdapterKind): string =>
  kind === 'z_image_control' || kind === 'anima_lllite' ? 'controlnet' : kind;

/** The backend owns kind support; count and Fill constraints are frontend policy. */
export const isControlKindSupportedForBase = (base: string, kind: ControlAdapterKind): boolean =>
  getArchitectureFeatures(base)?.control_kinds.includes(kind) ?? false;

/**
 * The kind to offer a layer whose kind `base` does not support: ControlNet where it is supported, otherwise the
 * base's one kind. Null when the base supports no control layers at all.
 */
export const getSuggestedControlKind = (base: string): ControlAdapterKind | null => {
  const supported = CONTROL_ADAPTER_KINDS.filter((kind) => isControlKindSupportedForBase(base, kind));
  return supported.includes('controlnet') ? 'controlnet' : (supported[0] ?? null);
};

/**
 * Whether a model can drive a control layer of `kind`, judged from the model alone. Model pickers filter with this
 * so they never offer what validation would reject.
 */
export const isControlModelUsableForKind = (model: ControlAdapterModelFacts, kind: ControlAdapterKind): boolean =>
  getControlModelUnusableReason(model, kind) === null;

/** Why a model cannot drive a control layer of `kind`, judged from the model alone; null when it can. */
export const getControlModelUnusableReason = (
  model: ControlAdapterModelFacts,
  kind: ControlAdapterKind
): ControlValidationReason | null => {
  if (model.type !== getControlAdapterModelType(kind)) {
    return 'incompatible_adapter';
  }
  if (kind === 'anima_lllite' && model.cond_in_channels !== 3) {
    // A control layer supplies only a control image; `anima_denoise` rejects a 4-channel (inpainting) adapter
    // without a mask, and an install that predates `cond_in_channels` may be either.
    return typeof model.cond_in_channels === 'number' ? 'lllite_inpaint_adapter' : 'lllite_channels_unknown';
  }
  return null;
};

export const getControlValidationReason = (params: {
  adapterModel: ControlAdapterModelFacts | null;
  beginEndStepPct: [number, number];
  controlLoraIndex: number;
  kind: ControlAdapterKind;
  mainBase: string;
  mainVariant?: string;
  weight: number;
  zImageControlIndex?: number;
  /** An earlier contributing Anima LLLite layer already applies this model; the denoiser applies each once. */
  lliteModelInUse?: boolean;
}): ControlValidationReason | null => {
  const {
    adapterModel,
    beginEndStepPct,
    controlLoraIndex,
    kind,
    lliteModelInUse = false,
    mainBase,
    mainVariant,
    weight,
    zImageControlIndex = 0,
  } = params;
  const capabilitiesLoaded = hasArchitectureCapabilities();
  // A kind the base cannot run comes first: switching kind is the fix, and it also resets or carries the model and
  // values, so asking for a model or valid values first would send the user the wrong way.
  if (capabilitiesLoaded && !isControlKindSupportedForBase(mainBase, kind)) {
    return getSuggestedControlKind(mainBase) ? 'switch_adapter_kind' : 'unsupported_adapter';
  }
  if (!areControlAdapterValuesValid(kind, weight, beginEndStepPct)) {
    return 'invalid_adapter_values';
  }
  if (!adapterModel) {
    return 'missing_model';
  }
  // Unavailable capability data is distinct from unsupported adapters.
  if (!capabilitiesLoaded) {
    return 'capabilities_unavailable';
  }
  if (adapterModel.base !== mainBase) {
    return 'incompatible_base';
  }
  const modelReason = getControlModelUnusableReason(adapterModel, kind);
  if (modelReason) {
    return modelReason;
  }
  if (kind === 'control_lora' && controlLoraIndex > 0) {
    return 'control_lora_limit';
  }
  if (kind === 'z_image_control' && zImageControlIndex > 0) {
    return 'z_image_control_limit';
  }
  if (kind === 'anima_lllite' && lliteModelInUse) {
    return 'duplicate_lllite_model';
  }
  if (kind === 'control_lora' && mainBase === 'flux' && mainVariant === 'dev_fill') {
    return 'flux_fill_control_lora';
  }
  return null;
};

/**
 * Validates control layers in generation order. Per-generation limits (one Control LoRA, one Z-Image control, each
 * LLLite model once) count only layers that passed, so an invalid layer never consumes a later layer's slot.
 */
export const createControlValidationSequence = (main: { base: string; variant?: string | null }) => {
  let controlLoraCount = 0;
  let zImageControlCount = 0;
  const lliteModelKeys = new Set<string>();

  return (layer: {
    adapterModel: (ControlAdapterModelFacts & { key: string }) | null;
    beginEndStepPct: [number, number];
    kind: ControlAdapterKind;
    weight: number;
  }): ControlValidationReason | null => {
    const { adapterModel, kind } = layer;
    const reason = getControlValidationReason({
      adapterModel,
      beginEndStepPct: layer.beginEndStepPct,
      controlLoraIndex: kind === 'control_lora' ? controlLoraCount : 0,
      kind,
      lliteModelInUse: kind === 'anima_lllite' && adapterModel !== null && lliteModelKeys.has(adapterModel.key),
      mainBase: main.base,
      mainVariant: main.variant ?? undefined,
      weight: layer.weight,
      zImageControlIndex: kind === 'z_image_control' ? zImageControlCount : 0,
    });
    if (reason) {
      return reason;
    }
    if (kind === 'control_lora') {
      controlLoraCount += 1;
    } else if (kind === 'z_image_control') {
      zImageControlCount += 1;
    } else if (kind === 'anima_lllite' && adapterModel) {
      lliteModelKeys.add(adapterModel.key);
    }
    return null;
  };
};

/** Whether `weight` and `beginEndStepPct` are valid for `kind`; validation, persisted-value repair and kind switches share it. */
export const areControlAdapterValuesValid = (
  kind: ControlAdapterKind,
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
