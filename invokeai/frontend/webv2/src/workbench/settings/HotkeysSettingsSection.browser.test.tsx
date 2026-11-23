import type * as projectsApi from '@workbench/projects/api';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { HotkeysSettingsSection } from './HotkeysSettingsSection';
import { DEFAULT_PREFERENCES, getWorkbenchPreferences, patchWorkbenchPreferences } from './store';

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
  it('rejects the stock Shift+N collision and saves an available New Raster Layer shortcut', async () => {
    await render();
    await act(() => search().fill('new raster layer'));
    await expect.element(page.getByText('New Raster Layer', { exact: true })).toBeVisible();
    await act(() => page.getByRole('button', { name: i18n.t('hotkeys.addHotkey'), exact: true }).click());
    await act(() => userEvent.keyboard('{Shift>}n{/Shift}'));
    await expect.element(conflictText('Fit Frame To Layers')).toBeVisible();
    const save = page.getByRole('button', { name: i18n.t('hotkeys.saveEdit'), exact: true });
    await expect.element(save).toBeDisabled();
    expect(getWorkbenchPreferences().customHotkeys['canvas.newRasterLayer']).toBeUndefined();

    await act(() => page.getByRole('button', { name: i18n.t('hotkeys.cancelEdit'), exact: true }).click());
    await act(() => page.getByRole('button', { name: i18n.t('hotkeys.addHotkey'), exact: true }).click());
    await act(() => userEvent.keyboard('{Alt>}{Shift>}n{/Shift}{/Alt}'));
    await expect.element(page.getByText(/^Conflicts with /)).not.toBeInTheDocument();
    await expect.element(save).toBeEnabled();
    await act(() => save.click());
    expect(getWorkbenchPreferences().customHotkeys['canvas.newRasterLayer']).toEqual(['shift+alt+n']);
  });

  // Recall Seed answers in Preview and Gallery. A Gallery binding on its key meets it through the second of those
  // widget types; a Canvas binding on the same key meets neither.
  it('flags a binding that shares its key with a multi-widget shortcut on any of that shortcut’s widgets', async () => {
    await patchWorkbenchPreferences({
      // Free S within Canvas so this fixture tests only cross-widget overlap with Recall Seed.
      customHotkeys: { 'canvas.tool.brush': ['s'], 'canvas.tool.spline': [], 'gallery.toggleStarredOnly': ['s'] },
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

  // Canvas binds T to its text tool, so a bare T toggles the left panel everywhere but the Canvas. Neither blocks the
  // other; both rows say so.
  it('notes on both rows when a widget key takes a global key while the widget has focus', async () => {
    await render();

    await act(() => search().fill('text tool'));
    await expect
      .element(
        page.getByText(i18n.t('hotkeys.shadows', { scope: 'Canvas', title: 'Toggle Left Panel' }), { exact: true })
      )
      .toBeVisible();

    await act(() => search().fill('toggle left panel'));
    await expect
      .element(
        page.getByText(i18n.t('hotkeys.shadowedBy', { scope: 'Canvas', title: 'Select Text Tool' }), { exact: true })
      )
      .toBeVisible();
    await expect.element(page.getByText(/^Conflicts with /)).not.toBeInTheDocument();
  });

  it('lists Clear Canvas on its own default shortcut, which meets no other default', async () => {
    await render();

    await act(() => search().fill('clear canvas'));
    await expect.element(page.getByText('Clear Canvas', { exact: true })).toBeVisible();
    await expect.element(page.getByText(/^Conflicts with /)).not.toBeInTheDocument();
  });
});
