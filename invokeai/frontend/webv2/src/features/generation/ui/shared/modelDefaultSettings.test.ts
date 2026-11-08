import type { ComponentModelConfig, MainModelConfig } from '@features/generation/core/types';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import {
  getDefaultGenerateSettings,
  getSettingsWithModelDefaults,
} from '@features/generation/core/baseGenerationPolicies';
import { describe, expect, it } from 'vitest';

import { getModelDefaultsPatch, settingsMatchModelDefaults } from './modelDefaultSettings';

seedArchitectureCapabilities();

const t5: ComponentModelConfig = { base: 'any', key: 't5', name: 'T5', type: 't5_encoder' };
const otherT5: ComponentModelConfig = { ...t5, key: 't5-fp8', name: 'T5 FP8' };
const flux: MainModelConfig = {
  base: 'flux',
  default_settings: { text_encoders: [t5.key] },
  key: 'flux',
  name: 'FLUX',
  type: 'main',
};

describe('Reset all to model defaults', () => {
  it('counts a text encoder other than the default as an override and resets it', () => {
    const settings = { ...getDefaultGenerateSettings(flux), t5EncoderModel: otherT5 };
    const models = [t5, otherT5];

    expect(settingsMatchModelDefaults(settings, getSettingsWithModelDefaults(settings, flux, models))).toBe(false);
    expect(getModelDefaultsPatch(settings, flux, models).t5EncoderModel).toEqual(t5);
    expect(getModelDefaultsPatch({ ...settings, t5EncoderModel: t5 }, flux, models)).not.toHaveProperty(
      't5EncoderModel'
    );
  });
});
