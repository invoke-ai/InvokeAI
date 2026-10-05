import i18n from 'i18next';
import { describe, expect, it } from 'vitest';

const enModules = import.meta.glob('../../../public/locales/en.json', { eager: true, import: 'default' });
const enGbModules = import.meta.glob('../../../public/locales/en-GB.json', { eager: true, import: 'default' });
const en = Object.values(enModules)[0] as { commandPalette: { title: string } };
const enGb = Object.values(enGbModules)[0] as Record<string, unknown>;

describe('command palette translations', () => {
  it('resolves command palette keys through the en fallback for en-GB', async () => {
    const instance = i18n.createInstance();
    await instance.init({
      lng: 'en-GB',
      fallbackLng: 'en',
      resources: {
        en: { translation: en },
        'en-GB': { translation: enGb },
      },
    });

    expect(instance.t('commandPalette.title')).toBe(en.commandPalette.title);
  });
});
