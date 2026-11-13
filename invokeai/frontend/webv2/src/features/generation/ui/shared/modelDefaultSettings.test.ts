import type { ComponentModelConfig, GenerateSettings, MainModelConfig } from '@features/generation/core/types';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { describe, expect, it } from 'vitest';

import {
  countModelDefaultOverrides,
  getModelDefaultsPatch,
  getModelDefaultSettings,
  settingsMatchModelDefaults,
} from './modelDefaultSettings';

seedArchitectureCapabilities();

const t5 = (key: string): ComponentModelConfig => ({ base: 'any', key, name: key, type: 't5_encoder' });
const clip: ComponentModelConfig = { base: 'any', key: 'clip', name: 'CLIP', type: 'clip_embed', variant: 'large' };
const defaultT5 = t5('t5-default');
const model: MainModelConfig = {
  base: 'flux',
  default_settings: { components: { clip_embed_model: 'missing', t5_encoder: defaultT5.key } },
  format: 'checkpoint',
  key: 'flux-model',
  name: 'FLUX',
  type: 'main',
};
const catalog = [defaultT5, t5('t5-mine'), clip];
const settingsWith = (overrides: Partial<GenerateSettings>): GenerateSettings => ({
  ...getDefaultGenerateSettings(model),
  ...overrides,
});

describe('reset to the model defaults, with default components', () => {
  it('replaces a selection with the default and keeps slots the model stores no usable default for', () => {
    const settings = settingsWith({ clipEmbedModel: clip, t5EncoderModel: t5('t5-mine') });
    const patch = getModelDefaultsPatch(settings, model, catalog);

    expect(patch.t5EncoderModel).toBe(defaultT5);
    // The CLIP default names a model that is not installed, so the current selection is not touched.
    expect(patch).not.toHaveProperty('clipEmbedModel');
  });

  it('counts a swapped default component as one override, and none once it is back', () => {
    const settings = settingsWith({ t5EncoderModel: t5('t5-mine') });
    const defaults = getModelDefaultSettings(settings, model, catalog);

    expect(countModelDefaultOverrides(settings, defaults)).toBe(1);
    expect(
      settingsMatchModelDefaults({ ...settings, ...getModelDefaultsPatch(settings, model, catalog) }, defaults)
    ).toBe(true);
  });
});
