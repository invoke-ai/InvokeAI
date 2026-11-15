import type { GenerateModelConfig, GenerateSettings, VaeModelConfig } from '@features/generation/core/types';
import type { TFunction } from 'i18next';

import { getLockedAspectRatioSettings } from '@features/generation/core/aspectRatioPresets';
import { isAspectRatioLocked } from '@features/generation/core/settings';

/** A setting the Model Manager can give a main model a default for. Width and height are one decision, size. */
export type ModelDefaultSetting =
  | 'cfgRescaleMultiplier'
  | 'cfgScale'
  | 'scheduler'
  | 'size'
  | 'steps'
  | 'vae'
  | 'vaePrecision';

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
