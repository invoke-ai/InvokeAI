import type { AnyModelDefaultSettings, ModelConfig, ModelTaxonomyType } from '@features/models/core/types';
import type { TFunction } from 'i18next';

import { loraDefaultSettingsSchema, mainDefaultSettingsSchema } from '@features/models/core/schemas';

/**
 * The pure policy behind per-model default settings: which model types get the
 * section, which fields each type shows, and how drafts validate. The section
 * component renders these specs; control descriptors here stay declarative so
 * this module needs no React.
 */

export type DefaultSettingsModel = Pick<ModelConfig, 'base' | 'default_settings' | 'key' | 'type'>;

const CONTROL_ADAPTER_TYPES = new Set(['controlnet', 't2i_adapter', 'control_lora']);

export const supportsDefaultSettings = (model: Pick<ModelConfig, 'type'>): boolean =>
  model.type === 'main' || model.type === 'lora' || CONTROL_ADAPTER_TYPES.has(model.type);

const SCHEDULERS = [
  'ddim',
  'ddpm',
  'deis',
  'deis_k',
  'dpmpp_2s',
  'dpmpp_2s_k',
  'dpmpp_2m',
  'dpmpp_2m_k',
  'dpmpp_2m_sde',
  'dpmpp_2m_sde_k',
  'dpmpp_3m',
  'dpmpp_3m_k',
  'dpmpp_sde',
  'dpmpp_sde_k',
  'euler',
  'euler_k',
  'euler_a',
  'heun',
  'heun_k',
  'kdpm_2',
  'kdpm_2_k',
  'kdpm_2_a',
  'kdpm_2_a_k',
  'lcm',
  'lms',
  'lms_k',
  'pndm',
  'tcd',
  'unipc',
  'unipc_k',
];

const PREPROCESSORS = [
  'canny_edge_detection',
  'color_map',
  'content_shuffle',
  'depth_anything_depth_estimation',
  'dw_openpose_detection',
  'hed_edge_detection',
  'lineart_anime_edge_detection',
  'lineart_edge_detection',
  'mediapipe_face_detection',
  'mlsd_detection',
  'normal_map',
  'pidi_edge_detection',
  'tile',
];

/** How the section renders a field's value; the component maps kind -> widget. */
export type DefaultSettingsControl =
  | { kind: 'number'; max?: number; min?: number; step?: number }
  | { kind: 'select'; options: readonly string[] }
  | { kind: 'combobox'; options: readonly string[] }
  | {
      kind: 'model';
      modelTypes: readonly ModelTaxonomyType[];
      /** Restrict candidates to the edited model's base. */
      sameBase: boolean;
      /** Placeholder shown for the backend's 'default' sentinel. */
      placeholderKey: string;
    };

export interface FieldSpec {
  key: keyof AnyModelDefaultSettings;
  labelKey: string;
  /**
   * Omitted for a boolean default, where the row's own enable switch already
   * expresses both states.
   */
  control?: DefaultSettingsControl;
  /** Value used when the toggle is switched on; also previewed while off. */
  defaultValue: unknown;
  /** Translation key for what applies while the toggle is off. */
  inheritLabelKey: string;
}

/**
 * Stores model weights as fp8 on the compute device, trading a little quality for VRAM.
 *
 * No body control: `_should_use_fp8` only acts on `fp8_storage is True`, so an explicit `false`
 * and an absent value behave identically. The row's enable switch already covers the two states
 * that differ, and a second switch inside the card would imply a distinction that does not exist.
 *
 * The backend gates availability in `_should_use_fp8`; `supportsFp8Storage` mirrors those
 * exclusions so the toggle is never offered where it would be silently ignored.
 */
const FP8_STORAGE_FIELD: FieldSpec = {
  defaultValue: true,
  inheritLabelKey: 'models.defaultFieldInherited.fp8Storage',
  key: 'fp8_storage',
  labelKey: 'models.defaultFields.fp8Storage',
};

const MAIN_FIELDS: FieldSpec[] = [
  {
    control: { kind: 'model', modelTypes: ['vae'], placeholderKey: 'models.defaultVae', sameBase: true },
    // The backend sentinel for "the compatible default VAE" — what legacy
    // writes when the toggle is on but no specific model is picked.
    defaultValue: 'default',
    inheritLabelKey: 'models.defaultFieldInherited.vae',
    key: 'vae',
    labelKey: 'models.defaultFields.vae',
  },
  {
    control: { kind: 'combobox', options: SCHEDULERS },
    defaultValue: 'euler_a',
    inheritLabelKey: 'models.defaultFieldInherited.scheduler',
    key: 'scheduler',
    labelKey: 'models.defaultFields.scheduler',
  },
  {
    control: { kind: 'number', max: 10000, min: 1 },
    defaultValue: 30,
    inheritLabelKey: 'models.defaultFieldInherited.steps',
    key: 'steps',
    labelKey: 'models.defaultFields.steps',
  },
  {
    control: { kind: 'number', max: 200, min: 1, step: 0.5 },
    defaultValue: 7,
    inheritLabelKey: 'models.defaultFieldInherited.cfgScale',
    key: 'cfg_scale',
    labelKey: 'models.defaultFields.cfgScale',
  },
  {
    control: { kind: 'number', max: 0.99, min: 0, step: 0.05 },
    defaultValue: 0,
    inheritLabelKey: 'models.defaultFieldInherited.cfgRescale',
    key: 'cfg_rescale_multiplier',
    labelKey: 'models.defaultFields.cfgRescale',
  },
  {
    control: { kind: 'number', max: 20, min: 1, step: 0.5 },
    defaultValue: 4,
    inheritLabelKey: 'models.defaultFieldInherited.guidance',
    key: 'guidance',
    labelKey: 'models.defaultFields.guidance',
  },
  {
    control: { kind: 'number', max: 8192, min: 64, step: 8 },
    defaultValue: 1024,
    inheritLabelKey: 'models.defaultFieldInherited.width',
    key: 'width',
    labelKey: 'models.defaultFields.width',
  },
  {
    control: { kind: 'number', max: 8192, min: 64, step: 8 },
    defaultValue: 1024,
    inheritLabelKey: 'models.defaultFieldInherited.height',
    key: 'height',
    labelKey: 'models.defaultFields.height',
  },
  {
    control: { kind: 'select', options: ['fp16', 'fp32'] },
    defaultValue: 'fp16',
    inheritLabelKey: 'models.defaultFieldInherited.vaePrecision',
    key: 'vae_precision',
    labelKey: 'models.defaultFields.vaePrecision',
  },
];

const LORA_FIELDS: FieldSpec[] = [
  {
    control: { kind: 'number', max: 10, min: -10, step: 0.05 },
    defaultValue: 0.75,
    inheritLabelKey: 'models.defaultFieldInherited.weight',
    key: 'weight',
    labelKey: 'models.defaultFields.weight',
  },
  {
    control: { kind: 'number', max: 10, min: -10, step: 0.05 },
    defaultValue: -1,
    inheritLabelKey: 'models.defaultFieldInherited.weightMin',
    key: 'weight_min',
    labelKey: 'models.defaultFields.weightMin',
  },
  {
    control: { kind: 'number', max: 10, min: -10, step: 0.05 },
    defaultValue: 2,
    inheritLabelKey: 'models.defaultFieldInherited.weightMax',
    key: 'weight_max',
    labelKey: 'models.defaultFields.weightMax',
  },
];

const CONTROL_ADAPTER_FIELDS: FieldSpec[] = [
  {
    control: { kind: 'select', options: PREPROCESSORS },
    defaultValue: 'canny_edge_detection',
    inheritLabelKey: 'models.defaultFieldInherited.preprocessor',
    key: 'preprocessor',
    labelKey: 'models.defaultFields.preprocessor',
  },
];

/**
 * Mirrors the backend's `_should_use_fp8` exclusions:
 * - Z-Image: diffusers' layerwise casting hits a dtype mismatch on skipped modules.
 * - LoRA / ControlLoRA: patched into a base model rather than run as their own forward pass, so
 *   the casting hooks would never fire.
 * VAEs are excluded too, but they have no default-settings section at all.
 */
export const supportsFp8Storage = (model: Pick<ModelConfig, 'base' | 'type'>): boolean => {
  if (model.base === 'z-image') {
    return false;
  }

  return model.type === 'main' || model.type === 'controlnet' || model.type === 't2i_adapter';
};

export const getFieldsForModel = (model: Pick<ModelConfig, 'base' | 'type'>): FieldSpec[] => {
  const fp8Fields = supportsFp8Storage(model) ? [FP8_STORAGE_FIELD] : [];

  if (model.type === 'main') {
    return [...MAIN_FIELDS, ...fp8Fields];
  }

  if (model.type === 'lora') {
    return LORA_FIELDS;
  }

  return [...CONTROL_ADAPTER_FIELDS, ...fp8Fields];
};

export const validateDefaults = (
  model: Pick<ModelConfig, 'type'>,
  settings: AnyModelDefaultSettings,
  t: TFunction
): string | null => {
  if (model.type === 'main') {
    const result = mainDefaultSettingsSchema.safeParse({
      cfgRescaleMultiplier: settings.cfg_rescale_multiplier ?? null,
      cfgScale: settings.cfg_scale ?? null,
      guidance: settings.guidance ?? null,
      height: settings.height ?? null,
      scheduler: settings.scheduler ?? null,
      steps: settings.steps ?? null,
      vae: settings.vae ?? null,
      vaePrecision: settings.vae_precision ?? null,
      width: settings.width ?? null,
    });

    return result.success ? null : (result.error.issues[0]?.message ?? t('models.invalidDefaultSettings'));
  }

  if (model.type === 'lora') {
    const result = loraDefaultSettingsSchema.safeParse({
      weight: settings.weight ?? null,
      weightMax: settings.weight_max ?? null,
      weightMin: settings.weight_min ?? null,
    });

    return result.success ? null : (result.error.issues[0]?.message ?? t('models.invalidDefaultSettings'));
  }

  return null;
};
