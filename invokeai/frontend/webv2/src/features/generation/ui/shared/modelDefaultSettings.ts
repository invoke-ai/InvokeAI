import type { GenerateModelConfig, GenerateSettings, VaeModelConfig } from '@features/generation/core/types';
import type { TFunction } from 'i18next';

import { formatAspectRatio, getLockedAspectRatioSettings } from '@features/generation/core/aspectRatioPresets';
import {
  getDefaultGenerateSettings,
  getGenerationDimensions,
  getGenerationUiPolicy,
  getSchedulerOptions,
  getSettingsWithModelDefaults,
  resolveModelDefaultVae,
} from '@features/generation/core/baseGenerationPolicies';
import { clampDimension, getActiveAspectRatio, isAspectRatioLocked } from '@features/generation/core/settings';

/** A setting the Model Manager can give a main model a default for. Width and height are one decision, size. */
export type ModelDefaultSetting =
  | 'cfgRescaleMultiplier'
  | 'cfgScale'
  | 'scheduler'
  | 'size'
  | 'steps'
  | 'vae'
  | 'vaePrecision';

/** What a reset is described by: the settings, plus the aspect ratio lock a size change can set or release. */
export type DescribedSetting = ModelDefaultSetting | 'aspectRatio';

/** In the order the form shows them: Size, then Render, then Advanced. */
const DESCRIBED_SETTINGS: readonly DescribedSetting[] = [
  'size',
  'aspectRatio',
  'steps',
  'cfgScale',
  'scheduler',
  'vae',
  'vaePrecision',
  'cfgRescaleMultiplier',
];

type OwnDefaults = NonNullable<Extract<GenerateModelConfig, { type: 'main' }>['default_settings']>;

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const getOwnDefaults = (model: GenerateModelConfig): OwnDefaults => (model.default_settings ?? {}) as OwnDefaults;

/** Where a reset takes its values from: the model's own defaults, the base's, or both. */
export type DefaultSource = 'base' | 'mixed' | 'model';

/**
 * Where a setting's default comes from, read the way the defaults are applied: guidance falls back to CFG and back,
 * only fp16 or fp32 is a precision, and a size with one side set takes the other from the base. An external model's
 * defaults hold at most a size. The VAE depends on the catalog: see `resolveModelDefaultVae`.
 */
const getSettingDefaultSource = (
  model: GenerateModelConfig,
  setting: Exclude<ModelDefaultSetting, 'vae'>
): DefaultSource => {
  const own = getOwnDefaults(model);

  switch (setting) {
    case 'size': {
      const sides = [own.width, own.height].filter(isFiniteNumber).length;

      return sides === 2 ? 'model' : sides === 1 ? 'mixed' : 'base';
    }
    case 'steps':
      return isFiniteNumber(own.steps) ? 'model' : 'base';
    case 'cfgScale':
      return isFiniteNumber(own.cfg_scale) || isFiniteNumber(own.guidance) ? 'model' : 'base';
    case 'cfgRescaleMultiplier':
      return isFiniteNumber(own.cfg_rescale_multiplier) ? 'model' : 'base';
    case 'scheduler':
      return typeof own.scheduler === 'string' && own.scheduler !== '' ? 'model' : 'base';
    case 'vaePrecision':
      return own.vae_precision === 'fp16' || own.vae_precision === 'fp32' ? 'model' : 'base';
  }
};

/** Whether the model's own defaults set the setting, even partly (one side of the size). */
export const doesModelSetDefault = (model: GenerateModelConfig, setting: Exclude<ModelDefaultSetting, 'vae'>) =>
  getSettingDefaultSource(model, setting) !== 'base';

export const getDefaultSource = (
  model: GenerateModelConfig,
  settings: readonly Exclude<ModelDefaultSetting, 'vae'>[]
): DefaultSource => {
  const sources = new Set(settings.map((setting) => getSettingDefaultSource(model, setting)));

  return sources.size === 1 ? [...sources][0]! : 'mixed';
};

/** Whether `vae` is the one the model's defaults name, so choosing it is no change from them. */
export const isModelDefaultVae = (model: GenerateModelConfig | undefined, vae: VaeModelConfig | null): boolean =>
  vae !== null && model !== undefined && (getOwnDefaults(model) as { vae?: unknown }).vae === vae.key;

/** Whether the form shows the setting for this family, so a value its graph ignores is neither applied nor told. */
const isShown = (setting: DescribedSetting, model: GenerateModelConfig): boolean => {
  const policy = getGenerationUiPolicy(model);

  switch (setting) {
    case 'scheduler':
      return policy.schedulerVisible;
    case 'cfgRescaleMultiplier':
      return policy.cfgRescaleVisible;
    case 'vaePrecision':
      return policy.vaePrecisionVisible;
    default:
      return true;
  }
};

type Dimensions = Pick<GenerateSettings, 'height' | 'width'>;

/** A whole default size: a lock stays on and takes the size's ratio, an unlocked ratio stays unlocked. */
export const getDefaultSizePatch = (
  settings: Pick<GenerateSettings, 'aspectRatioId' | 'aspectRatioIsLocked'>,
  { height, width }: Dimensions
): Partial<GenerateSettings> => {
  const ratio = height > 0 ? width / height : 1;

  return {
    ...(isAspectRatioLocked(settings)
      ? getLockedAspectRatioSettings(ratio)
      : { aspectRatioId: 'Free', aspectRatioIsLocked: false, aspectRatioValue: ratio }),
    height,
    width,
  };
};

/**
 * The model's size, as typing its values into the fields would set it. With only one side set, that side is typed: a
 * locked ratio carries the other side along, an unlocked one leaves it.
 */
const getOwnSizePatch = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  defaults: Dimensions
): Partial<GenerateSettings> => {
  const own = getOwnDefaults(model);
  const hasWidth = isFiniteNumber(own.width);

  if (hasWidth && isFiniteNumber(own.height)) {
    return getDefaultSizePatch(settings, defaults);
  }

  const isLocked = isAspectRatioLocked(settings);
  const ratio = getActiveAspectRatio(settings);
  const { grid } = getGenerationDimensions(model);

  return hasWidth
    ? { height: isLocked ? clampDimension(defaults.width / ratio, grid) : settings.height, width: defaults.width }
    : { height: defaults.height, width: isLocked ? clampDimension(defaults.height * ratio, grid) : settings.width };
};

/** The ratio a lock holds, by built-in id or value; one value for every unlocked state, whose stored ratio is unused. */
const getAspectRatioState = (settings: GenerateSettings): string =>
  !isAspectRatioLocked(settings)
    ? 'unlocked'
    : settings.aspectRatioId === 'Free'
      ? `Free:${settings.aspectRatioValue}`
      : settings.aspectRatioId;

const isSettingChanged = (setting: DescribedSetting, before: GenerateSettings, after: GenerateSettings): boolean => {
  switch (setting) {
    case 'size':
      return before.width !== after.width || before.height !== after.height;
    case 'aspectRatio':
      return getAspectRatioState(before) !== getAspectRatioState(after);
    case 'vae':
      return (before.vae?.key ?? null) !== (after.vae?.key ?? null);
    default:
      return !Object.is(before[setting], after[setting]);
  }
};

/**
 * What a change from `before` to `after` does to the settings the family shows, in form order. Both resets are told,
 * offered and toasted from this, so none of those can disagree with what the patch does.
 */
export const getChangedSettings = (
  before: GenerateSettings,
  after: GenerateSettings,
  model: GenerateModelConfig
): DescribedSetting[] =>
  DESCRIBED_SETTINGS.filter((setting) => isShown(setting, model) && isSettingChanged(setting, before, after));

export interface ApplyModelDefaultsPlan {
  /** Only the settings that change; empty when the model's defaults already hold. */
  patch: Partial<GenerateSettings>;
  changed: readonly DescribedSetting[];
  /** Shown settings the model sets a default for, changed or not. */
  defined: readonly ModelDefaultSetting[];
  /** Shown settings the model sets no default for, which applying leaves as they are. */
  unset: readonly ModelDefaultSetting[];
  /** The model names a VAE that is not installed or does not fit; the user's VAE stays. */
  isVaeMissing: boolean;
}

/**
 * Apply model defaults: only the values the model's own defaults set, as the legacy "Use default settings" did. A
 * setting the model leaves unset keeps its value, and a set one that already matches is left out of the patch.
 */
export const getApplyModelDefaultsPlan = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  vaeModels: readonly VaeModelConfig[]
): ApplyModelDefaultsPlan => {
  const defaults = getDefaultGenerateSettings(model);
  const defaultVae = resolveModelDefaultVae(model, settings, vaeModels);
  // Where the VAE has no field of its own, a default the model sets is still applied and told.
  const isVaeShown = getGenerationUiPolicy(model).sdVaeVisible || defaultVae.kind !== 'none';
  const patch: Partial<GenerateSettings> = {};
  const defined: ModelDefaultSetting[] = [];
  const unset: ModelDefaultSetting[] = [];

  for (const setting of DESCRIBED_SETTINGS) {
    if (setting === 'aspectRatio' || !isShown(setting, model) || (setting === 'vae' && !isVaeShown)) {
      continue;
    }

    const isSet = setting === 'vae' ? defaultVae.kind === 'apply' : doesModelSetDefault(model, setting);

    if (!isSet) {
      // A missing VAE is told apart from an unset one.
      if (!(setting === 'vae' && defaultVae.kind === 'missing')) {
        unset.push(setting);
      }
      continue;
    }

    defined.push(setting);
    const settingPatch: Partial<GenerateSettings> =
      setting === 'size'
        ? getOwnSizePatch(settings, model, defaults)
        : setting === 'vae'
          ? { vae: defaultVae.kind === 'apply' ? defaultVae.vae : null }
          : { [setting]: defaults[setting] };

    if (isSettingChanged(setting, settings, { ...settings, ...settingPatch })) {
      Object.assign(patch, settingPatch);
    }
  }

  return {
    changed: getChangedSettings(settings, { ...settings, ...patch }, model),
    defined,
    isVaeMissing: defaultVae.kind === 'missing',
    patch,
    unset,
  };
};

/**
 * Settings with every model-governed value reset: the model's where it sets one, the base's otherwise. The VAE changes
 * only to one the model's defaults can apply; otherwise the user's stays, since the base has no VAE to return to.
 */
export const getModelDefaultSettings = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  vaeModels: readonly VaeModelConfig[]
): GenerateSettings => {
  const nextSettings = getSettingsWithModelDefaults(settings, model);
  const defaultVae = resolveModelDefaultVae(model, settings, vaeModels);

  // The menu's policy for a default VAE that cannot apply: the user's VAE stays, and the menu says so.
  return defaultVae.kind === 'apply' ? { ...nextSettings, vae: defaultVae.vae } : nextSettings;
};

const MODEL_DEFAULT_VALUE_KEYS = [
  'aspectRatioId',
  'aspectRatioIsLocked',
  'aspectRatioValue',
  'cfgRescaleMultiplier',
  'cfgScale',
  'height',
  'modelKey',
  'scheduler',
  'steps',
  'vaePrecision',
  'width',
] as const satisfies readonly (keyof GenerateSettings)[];

/**
 * Reset everything to defaults, limited to the model-governed keys so prompts and other fields stay untouched.
 * Concepts are left as they are: the model already skips the ones it cannot load, and keeping them lets a compatible
 * model run them again.
 */
export const getModelDefaultsPatch = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  vaeModels: readonly VaeModelConfig[]
): Partial<GenerateSettings> => {
  const defaults = getModelDefaultSettings(settings, model, vaeModels);
  const patch: Partial<GenerateSettings> = { vae: defaults.vae };

  for (const key of MODEL_DEFAULT_VALUE_KEYS) {
    (patch as Record<string, unknown>)[key] = defaults[key];
  }

  return patch;
};

/** The setting's name as its field shows it: CFG or Guidance per family. */
const getSettingLabel = (setting: DescribedSetting, model: GenerateModelConfig, t: TFunction): string => {
  switch (setting) {
    case 'aspectRatio':
      return t('widgets.generate.aspectRatio');
    case 'cfgRescaleMultiplier':
      return t('widgets.generate.cfgRescale');
    case 'cfgScale':
      return getGenerationUiPolicy(model).guidanceLabel;
    case 'scheduler':
      return t('widgets.generate.scheduler');
    case 'size':
      return t('widgets.generate.size');
    case 'steps':
      return t('widgets.generate.steps');
    case 'vae':
      return t('widgets.generate.vae');
    case 'vaePrecision':
      return t('widgets.generate.vaePrecision');
  }
};

const formatSettingValue = (
  setting: DescribedSetting,
  settings: GenerateSettings,
  model: GenerateModelConfig,
  t: TFunction,
  locale: string | undefined
): string => {
  switch (setting) {
    // Digits as the size fields show them, without grouping.
    case 'size':
      return `${settings.width}×${settings.height}`;
    case 'aspectRatio':
      return !isAspectRatioLocked(settings)
        ? t('widgets.generate.aspectRatioFree')
        : settings.aspectRatioId === 'Free'
          ? formatAspectRatio(settings.aspectRatioValue, locale)
          : settings.aspectRatioId;
    case 'scheduler':
      return (
        getSchedulerOptions(model, settings.scheduler).find((option) => option.value === settings.scheduler)?.label ??
        settings.scheduler
      );
    case 'vae':
      return settings.vae?.name ?? t('widgets.generate.modelDefault');
    case 'vaePrecision':
      return settings.vaePrecision.toUpperCase();
    // Exactly as the fields show them, so two values that differ there never read the same here.
    default:
      return String(settings[setting]);
  }
};

// Constructing a ListFormat costs far more than formatting with one, and the menu formats on every open.
const listFormats = new Map<string, Intl.ListFormat>();
const formatList = (items: readonly string[], locale: string | undefined, type: 'conjunction' | 'unit'): string => {
  const key = `${locale ?? ''}\u0000${type}`;
  let format = listFormats.get(key);

  if (!format) {
    format = new Intl.ListFormat(locale, { style: type === 'unit' ? 'short' : 'long', type });
    listFormats.set(key, format);
  }

  return format.format(items);
};

/** The change as one list, such as "Steps 30 → 8, CFG 7 → 1"; empty when nothing changes. */
export const describeChanges = (
  changed: readonly DescribedSetting[],
  before: GenerateSettings,
  after: GenerateSettings,
  model: GenerateModelConfig,
  t: TFunction,
  locale: string | undefined
): string =>
  formatList(
    changed.map((setting) =>
      t('widgets.generate.modelDefaultsChange', {
        from: formatSettingValue(setting, before, model, t, locale),
        setting: getSettingLabel(setting, model, t),
        to: formatSettingValue(setting, after, model, t, locale),
      })
    ),
    locale,
    'unit'
  );

/** The settings Apply leaves as they are because the model sets no default, as one list; null for none. */
export const describeUnset = (
  unset: readonly ModelDefaultSetting[],
  model: GenerateModelConfig,
  t: TFunction,
  locale: string | undefined
): string | null =>
  unset.length > 0
    ? formatList(
        unset.map((setting) => getSettingLabel(setting, model, t)),
        locale,
        'conjunction'
      )
    : null;

/**
 * What a field's reset returns to, named honestly: the model's default when the model sets it, the base's (a main
 * model) when it does not, both for a size with one side set, or a plain default for an external model, whose
 * fallbacks belong to no base.
 */
export const getFieldResetLabel = (
  setting: Exclude<ModelDefaultSetting, 'vae'>,
  model: GenerateModelConfig | undefined,
  getBaseLabel: (base: string) => string,
  t: TFunction
): string => {
  const source = model ? getSettingDefaultSource(model, setting) : 'base';

  if (source === 'model') {
    return t('widgets.generate.resetToModelDefault');
  }

  if (model?.type !== 'main') {
    return t('widgets.generate.resetToDefault');
  }

  const base = getBaseLabel(model.base);

  return source === 'mixed'
    ? t('widgets.generate.resetToModelAndBaseDefault', { base })
    : t('widgets.generate.resetToBaseDefault', { base });
};
