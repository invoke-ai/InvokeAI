import type * as projectsApi from '@workbench/projects/api';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { HotkeysSettingsSection } from './HotkeysSettingsSection';
import { DEFAULT_PREFERENCES, patchWorkbenchPreferences } from './store';

vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));

const translations = await fetch('/locales/en.json').then((response) => response.json());
const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  fallbackLng: 'en',
  resources: { en: { translation: translations } },
  interpolation: { escapeValue: false },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

const render = () =>
  act(() => {
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <HotkeysSettingsSection />
        </I18nextProvider>
      </ChakraProvider>
    );
  });

const search = () => page.getByRole('textbox', { name: i18n.t('hotkeys.searchPlaceholder'), exact: true });
const conflictText = (title: string) => page.getByText(i18n.t('hotkeys.conflictsWith', { title }), { exact: true });

beforeEach(async () => {
  await page.viewport(1280, 900);
  accountLifecycle.activate('hotkeys-settings-test');
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  host = document.createElement('div');
  host.style.height = '800px';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  accountLifecycle.invalidate();
});

describe('HotkeysSettingsSection conflicts', () => {
  // Recall Seed answers in Preview and Gallery. A Gallery binding on its key meets it through the second of those
  // widget types; a Canvas binding on the same key meets neither.
  it('flags a binding that shares its key with a multi-widget shortcut on any of that shortcut’s widgets', async () => {
    await patchWorkbenchPreferences({
      customHotkeys: { 'canvas.selectBrushTool': ['s'], 'gallery.toggleStarredOnly': ['s'] },
    });
    await render();

    await act(() => search().fill('starred'));
    await expect.element(page.getByText('Toggle Starred Only', { exact: true })).toBeVisible();
    await expect.element(conflictText('Recall Seed')).toBeVisible();

    await act(() => search().fill('brush'));
    await expect.element(page.getByText('Select Brush Tool', { exact: true })).toBeVisible();
    await expect.element(conflictText('Recall Seed')).not.toBeInTheDocument();
    await expect.element(page.getByText(/^Conflicts with /)).not.toBeInTheDocument();
  });
});
