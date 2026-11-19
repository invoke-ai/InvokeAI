import type { GenerateSettings, MainModelConfig, VaeModelConfig } from '@features/generation/core/types';
import type { TFunction } from 'i18next';

import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import i18n from 'i18next';
import { beforeAll, describe, expect, it } from 'vitest';

import {
  describeChanges,
  describeUnset,
  getApplyModelDefaultsPlan,
  getChangedSettings,
  getModelDefaultsPatch,
} from './modelDefaultsReset';

seedArchitectureCapabilities();

const englishCatalog = Object.values(
  import.meta.glob('../../../../../public/locales/en.json', { eager: true, import: 'default' })
)[0] as Record<string, unknown>;
const instance = i18n.createInstance();
const t = ((key: string, values?: Record<string, unknown>) => instance.t(key, values)) as TFunction;

beforeAll(async () => {
  await instance.init({
    initAsync: false,
    interpolation: { escapeValue: false },
    lng: 'en',
    resources: { en: { translation: englishCatalog } },
  });
});

const sdxl = (defaults: MainModelConfig['default_settings'] = null): MainModelConfig => ({
  base: 'sdxl',
  default_settings: defaults,
  key: 'sdxl-model',
  name: 'Juggernaut',
  type: 'main',
});
const fixVae: VaeModelConfig = { base: 'sdxl', key: 'sdxl-vae-fix', name: 'SDXL VAE fp16 fix', type: 'vae' };
const fluxVae: VaeModelConfig = { base: 'flux', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' };
const customVae: VaeModelConfig = { base: 'sdxl', key: 'custom-vae', name: 'Custom VAE', type: 'vae' };

const userSettings = (overrides: Partial<GenerateSettings> = {}): GenerateSettings => ({
  ...getDefaultGenerateSettings(sdxl()),
  aspectRatioId: 'Free',
  aspectRatioIsLocked: false,
  aspectRatioValue: 1216 / 832,
  cfgScale: 4,
  height: 832,
  scheduler: 'dpmpp_2m_k',
  steps: 12,
  vae: customVae,
  vaePrecision: 'fp16',
  width: 1216,
  ...overrides,
});

const after = (settings: GenerateSettings, patch: Partial<GenerateSettings>) => ({ ...settings, ...patch });

describe('getApplyModelDefaultsPlan', () => {
  it('changes only what the model sets: steps and CFG, leaving the size and everything else alone', () => {
    const model = sdxl({ cfg_scale: 1, steps: 8 });
    const plan = getApplyModelDefaultsPlan(userSettings(), model, [fixVae, customVae]);

    expect(plan.patch).toEqual({ cfgScale: 1, steps: 8 });
    expect(plan.changed).toEqual(['steps', 'cfgScale']);
    expect(plan.isVaeMissing).toBe(false);
    expect(plan.unset).toEqual(['size', 'scheduler', 'vae', 'vaePrecision']);
  });

  it('types a lone default width into the field: an unlocked height stays, a locked one follows the ratio', () => {
    const model = sdxl({ width: 896 });

    expect(getApplyModelDefaultsPlan(userSettings(), model, []).patch).toEqual({ height: 832, width: 896 });
    expect(
      getApplyModelDefaultsPlan(
        userSettings({ aspectRatioId: '16:9', aspectRatioValue: 16 / 9, height: 576, width: 1024 }),
        model,
        []
      ).patch
    ).toEqual({ height: 504, width: 896 });
  });

  it('types a lone default height the same way', () => {
    expect(getApplyModelDefaultsPlan(userSettings(), sdxl({ height: 1152 }), []).patch).toEqual({
      height: 1152,
      width: 1216,
    });
  });

  it('applies a whole default size with the ratio following it, keeping the lock as it was', () => {
    const model = sdxl({ height: 1152, width: 768 });

    expect(getApplyModelDefaultsPlan(userSettings(), model, []).patch).toEqual({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: false,
      aspectRatioValue: 768 / 1152,
      height: 1152,
      width: 768,
    });
    expect(
      getApplyModelDefaultsPlan(userSettings({ aspectRatioId: '1:1', aspectRatioValue: 1 }), model, []).patch
    ).toMatchObject({ aspectRatioId: '2:3', aspectRatioIsLocked: true, height: 1152, width: 768 });
  });

  it('has nothing to apply when the model sets no defaults, however far the settings are from the base', () => {
    const plan = getApplyModelDefaultsPlan(userSettings(), sdxl(), [fixVae]);

    expect(plan.patch).toEqual({});
    expect(plan.defined).toEqual([]);
  });

  it('has nothing to apply once the values the model sets hold, whatever the rest is', () => {
    const plan = getApplyModelDefaultsPlan(userSettings({ steps: 8 }), sdxl({ steps: 8 }), []);

    expect(plan.patch).toEqual({});
    expect(plan.defined).toEqual(['steps']);
  });

  it("applies the model's VAE when it sets one, the bundled one for the 'default' sentinel, and none otherwise", () => {
    expect(getApplyModelDefaultsPlan(userSettings(), sdxl({ vae: fixVae.key }), [fixVae]).patch).toEqual({
      vae: fixVae,
    });
    expect(getApplyModelDefaultsPlan(userSettings(), sdxl({ vae: 'default' }), [fixVae]).patch).toEqual({
      vae: null,
    });
    expect(
      getApplyModelDefaultsPlan(userSettings({ vae: { ...fixVae } }), sdxl({ vae: fixVae.key }), [fixVae]).patch
    ).toEqual({});
    expect(getApplyModelDefaultsPlan(userSettings(), sdxl({ steps: 30 }), [fixVae]).patch).not.toHaveProperty('vae');
  });

  it("applies the bundled VAE for the 'default' sentinel where the VAE is an optional component", () => {
    const sd3: MainModelConfig = { ...sdxl({ vae: 'default' }), base: 'sd-3', format: 'diffusers' };
    const sd3Vae: VaeModelConfig = { base: 'sd-3', key: 'sd3-vae', name: 'SD3 VAE', type: 'vae' };
    const settings = { ...getDefaultGenerateSettings(sd3), vae: sd3Vae };
    const plan = getApplyModelDefaultsPlan(settings, sd3, [sd3Vae]);

    expect(plan.patch).toEqual({ vae: null });
    expect(describeChanges(plan.changed, settings, after(settings, plan.patch), sd3, t, 'en')).toBe(
      'VAE SD3 VAE → Bundled VAE'
    );
    expect(getApplyModelDefaultsPlan({ ...settings, vae: null }, sd3, [sd3Vae]).defined).toEqual(['vae']);
  });

  it("keeps the user's VAE when the model names one that is not installed, and says so", () => {
    const model = sdxl({ steps: 8, vae: 'uninstalled-vae' });
    const plan = getApplyModelDefaultsPlan(userSettings(), model, [fixVae]);

    expect(plan.patch).toEqual({ steps: 8 });
    expect(plan.isVaeMissing).toBe(true);
    expect(describeUnset(plan.unset, model, t, 'en')).toBe('Size, CFG, Scheduler, and VAE precision');
    expect(after(userSettings(), getModelDefaultsPatch(userSettings(), model, [fixVae])).vae).toBe(customVae);
  });

  it("never clears a family's required VAE for the 'default' sentinel, which has no bundled VAE there", () => {
    const flux: MainModelConfig = { ...sdxl({ steps: 20, vae: 'default' }), base: 'flux' };
    const settings = { ...getDefaultGenerateSettings(flux), steps: 28, vae: fluxVae };
    const plan = getApplyModelDefaultsPlan(settings, flux, [fluxVae]);

    expect(plan.patch).toEqual({ steps: 20 });
    expect(plan.unset).not.toContain('vae');
    expect(after(settings, getModelDefaultsPatch(settings, flux, [fluxVae])).vae).toBe(fluxVae);
    expect(
      getApplyModelDefaultsPlan({ ...settings, vae: null }, { ...flux, default_settings: { vae: fluxVae.key } }, [
        fluxVae,
      ]).patch
    ).toEqual({ vae: fluxVae });
  });

  it('leaves out a setting the form does not show for the family, such as a scheduler its graph ignores', () => {
    const qwen: MainModelConfig = { ...sdxl({ scheduler: 'dpmpp_2m_k', steps: 20 }), base: 'qwen-image' };
    const plan = getApplyModelDefaultsPlan(userSettings({ scheduler: 'euler' }), qwen, []);

    expect(plan.patch).toEqual({ steps: 20 });
    expect(plan.unset).not.toContain('scheduler');
  });
});

describe('getModelDefaultsPatch (reset everything)', () => {
  it('tells a ratio the reset locks, even when the size stays, and leaves out what the family hides', () => {
    const model = sdxl();
    const settings = userSettings({
      cfgRescaleMultiplier: 0.7,
      cfgScale: 7,
      height: 1024,
      scheduler: 'euler_a',
      steps: 30,
      vaePrecision: 'fp32',
      width: 1024,
    });
    const next = after(settings, getModelDefaultsPatch(settings, model, []));
    const changed = getChangedSettings(settings, next, model);

    expect(changed).toEqual(['aspectRatio']);
    expect(describeChanges(changed, settings, next, model, t, 'en')).toBe('Aspect ratio Free → 1:1');
  });

  it("still fills what the model leaves unset from the base, size included, and keeps the user's VAE", () => {
    const settings = userSettings();
    const patch = getModelDefaultsPatch(settings, sdxl({ steps: 8 }), [fixVae]);

    expect(patch).toMatchObject({ cfgScale: 7, height: 1024, scheduler: 'euler_a', steps: 8, width: 1024 });
    expect(after(settings, patch).vae).toBe(customVae);
  });
});

describe('describeChanges', () => {
  it('names each change as its field does, from the old value to the new one', () => {
    const model = sdxl({
      height: 1344,
      scheduler: 'euler_a',
      steps: 8,
      vae: fixVae.key,
      vae_precision: 'fp32',
      width: 768,
    });
    const settings = userSettings();
    const plan = getApplyModelDefaultsPlan(settings, model, [fixVae]);

    expect(describeChanges(plan.changed, settings, after(settings, plan.patch), model, t, 'en')).toBe(
      'Size 1216×832 → 768×1344, Steps 12 → 8, Scheduler DPM++ 2M Karras → Euler Ancestral, ' +
        'VAE Custom VAE → SDXL VAE fp16 fix, VAE precision FP16 → FP32'
    );
  });

  it('calls the guidance setting what the family calls it', () => {
    const flux: MainModelConfig = { ...sdxl({ guidance: 2.5 }), base: 'flux' };
    const settings = { ...getDefaultGenerateSettings(flux), cfgScale: 4 };
    const plan = getApplyModelDefaultsPlan(settings, flux, []);

    expect(describeChanges(plan.changed, settings, after(settings, plan.patch), flux, t, 'en')).toBe(
      'Guidance 4 → 2.5'
    );
    expect(describeUnset(plan.unset, flux, t, 'en')).toBe('Size, Steps, and Scheduler');
  });

  it('writes numbers exactly as the fields show them, so differing values never read the same', () => {
    const model = sdxl({ cfg_scale: 7.125 });
    const settings = userSettings({ cfgScale: 7.1254 });
    const plan = getApplyModelDefaultsPlan(settings, model, []);

    expect(describeChanges(plan.changed, settings, after(settings, plan.patch), model, t, 'en')).toBe(
      'CFG 7.1254 → 7.125'
    );
  });
});
