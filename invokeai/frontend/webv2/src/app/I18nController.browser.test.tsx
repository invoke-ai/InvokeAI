import type * as Languages from '@platform/i18n/languages';
import type { WorkbenchLanguage } from '@platform/i18n/languages';

import i18n from '@platform/i18n/client';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

const settings = vi.hoisted(() => ({
  snapshot: { status: 'loading', preferences: { language: 'ar' as WorkbenchLanguage } },
  listeners: new Set<() => void>(),
  readyArabic: false,
}));
vi.mock('@workbench/settings/store', () => ({
  getWorkbenchSettingsSnapshot: () => settings.snapshot,
  subscribeWorkbenchSettings: (listener: () => void) => {
    settings.listeners.add(listener);
    return () => settings.listeners.delete(listener);
  },
}));
vi.mock('@platform/i18n/languages', async (importOriginal) => {
  const original = await importOriginal<typeof Languages>();
  return {
    ...original,
    getEffectiveWorkbenchLanguage: (language: WorkbenchLanguage) =>
      settings.readyArabic && language === 'ar'
        ? 'ar'
        : original.getEffectiveWorkbenchLanguage(language, {
            en: { translated: 1, ready: true },
            fr: { translated: 1 },
          }),
  };
});
vi.mock('@platform/i18n/client', async () => {
  const { createInstance } = await import('i18next');
  const client = createInstance();
  await client.init({
    lng: 'en',
    fallbackLng: 'en',
    resources: { en: { translation: {} }, 'en-GB': { translation: {} } },
  });
  return { default: client };
});

import { I18nController } from './I18nController';

let host: HTMLDivElement;
let root: ReturnType<typeof createRoot>;
beforeEach(async () => {
  host = document.createElement('div');
  root = createRoot(host);
  document.body.append(host);
  settings.snapshot = { status: 'loading', preferences: { language: 'ar' } };
  settings.readyArabic = false;
  await i18n.changeLanguage('en');
});
const change = async (language: WorkbenchLanguage) => {
  settings.snapshot = { status: 'ready', preferences: { language } };
  await act(async () => {
    for (const listener of settings.listeners) {
      listener();
    }
    await Promise.resolve();
  });
};
afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  localStorage.removeItem('invokeai:v7:webv2:language');
  document.documentElement.lang = 'en';
  document.documentElement.dir = 'ltr';
  document.body.dir = 'ltr';
});

it('hydrates saved unavailable content as English and keeps the requested preference through language changes', async () => {
  document.documentElement.lang = 'en';
  document.documentElement.dir = 'ltr';
  await act(() => root.render(<I18nController />));
  await change('ar');
  expect(settings.snapshot.preferences.language).toBe('ar');
  expect(i18n.language).toBe('en');
  expect(document.documentElement.lang).toBe('en');
  expect(document.documentElement.dir).toBe('ltr');
  expect(document.body.dir).toBe('ltr');
  expect(localStorage.getItem('invokeai:v7:webv2:language')).toBe('ar');
  await change('en-GB');
  await change('fr');
  await expect.poll(() => i18n.language).toBe('en');
  expect(document.documentElement.lang).toBe('en');
  expect(settings.snapshot.preferences.language).toBe('fr');
});

it('applies the latest effective language after an overlapping deferred change', async () => {
  await act(() => root.render(<I18nController />));
  const original = i18n.changeLanguage.bind(i18n);
  let release = () => {};
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  const delayed = vi.spyOn(i18n, 'changeLanguage').mockImplementationOnce(async (language) => {
    await pending;
    return original(language);
  });
  try {
    await change('en-GB');
    await change('fr');
    release();
    await delayed.mock.results[0]?.value;
    await expect.poll(() => i18n.language).toBe('en');
    await expect.poll(() => document.documentElement.lang).toBe('en');
    expect(document.documentElement.dir).toBe('ltr');
    expect(document.body.dir).toBe('ltr');
    expect(localStorage.getItem('invokeai:v7:webv2:language')).toBe('fr');
    expect(settings.snapshot.preferences.language).toBe('fr');
  } finally {
    release();
    delayed.mockRestore();
  }
});

it('hydrates an explicitly ready Arabic fixture with matching content and RTL direction', async () => {
  settings.readyArabic = true;
  i18n.addResourceBundle('ar', 'translation', { fixture: 'مرحبا' });
  await act(() => root.render(<I18nController />));
  await change('ar');
  await expect.poll(() => document.documentElement.lang).toBe('ar');
  expect(i18n.t('fixture')).toBe('مرحبا');
  expect(document.documentElement.dir).toBe('rtl');
  expect(document.body.dir).toBe('rtl');
  expect(settings.snapshot.preferences.language).toBe('ar');
});

it('writes the saved language only when the preference changes', async () => {
  const writes = vi.spyOn(Storage.prototype, 'setItem');
  try {
    await act(() => root.render(<I18nController />));
    await change('ar');
    await change('ar');
    await change('fr');
    await change('fr');
    expect(writes.mock.calls.filter(([key]) => key === 'invokeai:v7:webv2:language')).toEqual([
      ['invokeai:v7:webv2:language', 'ar'],
      ['invokeai:v7:webv2:language', 'fr'],
    ]);
  } finally {
    writes.mockRestore();
  }
});
