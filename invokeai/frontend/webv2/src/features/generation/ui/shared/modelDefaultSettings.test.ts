import type { ExternalImageGeneratorModelConfig, MainModelConfig } from '@features/generation/core/types';
import type { TFunction } from 'i18next';

import i18n from 'i18next';
import { beforeAll, describe, expect, it } from 'vitest';

import { getDefaultSource, getFieldResetLabel } from './modelDefaultSettings';

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

describe('reset labels', () => {
  const getBaseLabel = (base: string) => (base === 'sdxl' ? 'SDXL' : base);
  const external: ExternalImageGeneratorModelConfig = {
    base: 'external',
    format: 'external_api',
    key: 'gpt-image',
    name: 'GPT Image',
    type: 'external_image_generator',
  };

  it("names the model's default where it sets one and the base's where it does not", () => {
    const model = sdxl({ steps: 8 });

    expect(getFieldResetLabel('steps', model, getBaseLabel, t)).toBe('Reset to model default');
    expect(getFieldResetLabel('size', model, getBaseLabel, t)).toBe('Reset to SDXL default');
    expect(getFieldResetLabel('size', sdxl({ width: 896 }), getBaseLabel, t)).toBe('Reset to model and SDXL default');
    expect(getFieldResetLabel('size', external, getBaseLabel, t)).toBe('Reset to default');
  });

  it('tells a section whose settings come from both sources apart', () => {
    const model = sdxl({ steps: 8 });

    expect(getDefaultSource(model, ['steps'])).toBe('model');
    expect(getDefaultSource(model, ['steps', 'cfgScale'])).toBe('mixed');
    expect(getDefaultSource(model, ['size'])).toBe('base');
  });
});
