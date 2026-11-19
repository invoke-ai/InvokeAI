import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';
import type { TFunction } from 'i18next';

import { formatAspectRatio } from '@features/generation/core/aspectRatioPresets';
import {
  getDefaultGenerateSettings,
  getComponentSectionPolicy,
  getGenerationDimensions,
  getGenerationUiPolicy,
  getModelDefaultComponents,
  getSchedulerOptions,
  getSettingsWithModelDefaults,
  resolveModelDefaultVae,
} from '@features/generation/core/baseGenerationPolicies';
import { DEFAULT_COMPONENT_SLOTS, type DefaultComponentSlot } from '@features/generation/core/componentDefaults';
import {
  clampDimension,
  getActiveAspectRatio,
  isAspectRatioLocked,
  isVaeModelConfig,
} from '@features/generation/core/settings';

import {
  doesModelSetDefault,
  getDefaultSizePatch,
  getDefaultSource,
  type ModelDefaultSetting,
} from './modelDefaultSettings';

// The Generate reset menu's plan and wording. Only that menu reads them, and it loads with the menu, off the editor's
// startup; the fields' reset labels read `modelDefaultSettings`.

/** A component slot a main model can store a default model for. */
type ComponentSetting = DefaultComponentSlot['setting'];

/** What a reset is described by: the settings, the aspect ratio lock a size change can set or release, components. */
export type DescribedSetting = ModelDefaultSetting | 'aspectRatio' | ComponentSetting;

const COMPONENT_SETTINGS: readonly ComponentSetting[] = DEFAULT_COMPONENT_SLOTS.map(({ setting }) => setting);

const isComponentSetting = (setting: DescribedSetting): setting is ComponentSetting =>
  (COMPONENT_SETTINGS as readonly DescribedSetting[]).includes(setting);

/** In the order the form shows them: Size, then Render, then Components, then Advanced. */
const DESCRIBED_SETTINGS: readonly DescribedSetting[] = [
  'size',
  'aspectRatio',
  'steps',
  'cfgScale',
  'scheduler',
  'vae',
  ...COMPONENT_SETTINGS,
  'vaePrecision',
  'cfgRescaleMultiplier',
];

/**
 * Whether the form shows the setting for this family, so a value its graph ignores is neither applied nor told. A
 * component changes only to a default for a slot the family offers, so its change is always told.
 */
const isShown = (setting: DescribedSetting, model: GenerateModelConfig): boolean => {
  if (isComponentSetting(setting)) {
    return true;
  }

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

/**
 * The model's size, as typing its values into the fields would set it. With only one side set, that side is typed: a
 * locked ratio carries the other side along, an unlocked one leaves it.
 */
const getOwnSizePatch = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  defaults: Pick<GenerateSettings, 'height' | 'width'>
): Partial<GenerateSettings> => {
  if (getDefaultSource(model, ['size']) === 'model') {
    return getDefaultSizePatch(settings, defaults);
  }

  // The model sets one side.
  const hasWidth = Number.isFinite((model.default_settings as { width?: unknown } | null | undefined)?.width);

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
      return isComponentSetting(setting)
        ? (before[setting]?.key ?? null) !== (after[setting]?.key ?? null)
        : !Object.is(before[setting], after[setting]);
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
  /** Shown settings and component slots the model sets a usable default for, changed or not. */
  defined: readonly (ModelDefaultSetting | ComponentSetting)[];
  /** Shown settings the model sets no default for, which applying leaves as they are. */
  unset: readonly ModelDefaultSetting[];
  /** The model names a VAE that is not installed or does not fit; the user's VAE stays. */
  isVaeMissing: boolean;
}

/**
 * Apply model defaults: only the values the model's own defaults set, as the legacy "Use default settings" did. A
 * setting the model leaves unset keeps its value, and a set one that already matches is left out of the patch. A
 * default component replaces the slot's selection; one that is not installed or does not fit is skipped, and the
 * slots the model stores no default for are not listed as unset.
 */
export const getApplyModelDefaultsPlan = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  models: readonly ModelConfig[]
): ApplyModelDefaultsPlan => {
  const defaults = getDefaultGenerateSettings(model);
  const components = getModelDefaultComponents(model, settings, models, { replace: true });
  const withComponents = { ...settings, ...components };
  // A default component source can bundle the VAE, so the VAE is judged beside the default components.
  const defaultVae = resolveModelDefaultVae(model, withComponents, models.filter(isVaeModelConfig));
  // Where the VAE has no field of its own, a default the model sets is still applied and told.
  const isVaeShown = getGenerationUiPolicy(model).sdVaeVisible || defaultVae.kind !== 'none';
  const patch: Partial<GenerateSettings> = {};
  const defined: (ModelDefaultSetting | ComponentSetting)[] = [];
  const unset: ModelDefaultSetting[] = [];

  for (const setting of DESCRIBED_SETTINGS) {
    if (isComponentSetting(setting)) {
      // A cleared slot (`null`: the default encoder closed it) is no default of the model's, only a change.
      if (setting in components) {
        if (components[setting]) {
          defined.push(setting);
        }

        if (isSettingChanged(setting, settings, withComponents)) {
          Object.assign(patch, { [setting]: components[setting] });
        }
      }
      continue;
    }

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
 * Settings with every model-governed value reset: the model's where it sets one, the base's otherwise. The VAE and
 * each component change only to a default the model can apply; otherwise the user's stays, since the base has none
 * to return to.
 */
export const getModelDefaultSettings = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  models: readonly ModelConfig[]
): GenerateSettings => {
  const withDefaults = getSettingsWithModelDefaults(settings, model);
  // Components first: a default component source can bundle the VAE.
  const nextSettings = {
    ...withDefaults,
    ...getModelDefaultComponents(model, withDefaults, models, { replace: true }),
  };
  const defaultVae = resolveModelDefaultVae(model, nextSettings, models.filter(isVaeModelConfig));

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
  models: readonly ModelConfig[]
): Partial<GenerateSettings> => {
  const defaults = getModelDefaultSettings(settings, model, models);
  const patch: Partial<GenerateSettings> = { vae: defaults.vae };

  for (const key of MODEL_DEFAULT_VALUE_KEYS) {
    (patch as Record<string, unknown>)[key] = defaults[key];
  }

  // Only the component slots a default changes; the others keep their selection untouched.
  for (const setting of COMPONENT_SETTINGS) {
    if (defaults[setting] !== settings[setting]) {
      Object.assign(patch, { [setting]: defaults[setting] });
    }
  }

  return patch;
};

/**
 * A component slot's name as the Components section shows it. A slot can depend on another (the vision tower on a GGUF
 * encoder), so it is looked up beside the settings the change leads to.
 */
const getComponentLabel = (
  setting: ComponentSetting,
  model: GenerateModelConfig,
  settings: GenerateSettings,
  t: TFunction
): string =>
  getComponentSectionPolicy(model, settings).slots.find((slot) => slot.key === setting)?.label ??
  t('widgets.generate.components');

/** The setting's name as its field shows it: CFG or Guidance per family. */
const getSettingLabel = (
  setting: Exclude<DescribedSetting, ComponentSetting>,
  model: GenerateModelConfig,
  t: TFunction
): string => {
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
      return settings.vae?.name ?? t('widgets.generate.bundledVae');
    case 'vaePrecision':
      return settings.vaePrecision.toUpperCase();
    // Exactly as the fields show them, so two values that differ there never read the same here.
    default:
      return isComponentSetting(setting) ? (settings[setting]?.name ?? t('common.none')) : String(settings[setting]);
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
        setting: isComponentSetting(setting)
          ? getComponentLabel(setting, model, after, t)
          : getSettingLabel(setting, model, t),
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
