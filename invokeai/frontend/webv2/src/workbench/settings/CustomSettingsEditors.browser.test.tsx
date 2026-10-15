import type { SettingDefinition } from '@platform/ui/settings/contracts';
import type * as projectsApi from '@workbench/projects/api';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system, THEMES } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { appearanceSettings, developerSettings, hotkeysSettings, workspaceSettings } from './applicationContributions';
import CustomSettingField from './CustomSettingsEditors';
import { DEFAULT_PREFERENCES, getWorkbenchPreferences, patchWorkbenchPreferences } from './store';

vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  resources: { en: { translation: { settingsDialog: { changedSetting: '{{setting}} differs from its default' } } } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let host: HTMLDivElement;
let root: Root;
const render = async (field: SettingDefinition, onReveal?: (sectionId: string, entryId: string) => void) => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <CustomSettingField field={field} surface="dialog" onReveal={onReveal} />
        </I18nextProvider>
      </ChakraProvider>
    )
  );
};
const modifiedIndicators = () => host.querySelectorAll('[role="img"][aria-label$="differs from its default"]');

beforeEach(async () => {
  accountLifecycle.activate('custom-settings-test');
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

describe('custom setting modified indicators', () => {
  it('marks a changed theme and clears the marker when the default theme is selected', async () => {
    await render(appearanceSettings.fields.find((field) => field.id === 'themeId')!);
    expect(modifiedIndicators()).toHaveLength(0);
    const alternate = THEMES.find((theme) => theme.id !== DEFAULT_PREFERENCES.themeId)!;
    const selectTheme = async (label: string) => {
      const button = Array.from(host.querySelectorAll('button')).find((candidate) =>
        candidate.textContent?.startsWith(label)
      )!;
      await act(async () => {
        button.click();
        await Promise.resolve();
      });
    };
    await selectTheme(alternate.label);
    expect(getWorkbenchPreferences().themeId).toBe(alternate.id);
    expect(modifiedIndicators()).toHaveLength(1);
    await selectTheme(THEMES.find((theme) => theme.id === DEFAULT_PREFERENCES.themeId)!.label);
    expect(modifiedIndicators()).toHaveLength(0);
  });

  it('marks a changed namespace selection and clears the marker after restoring that checkbox', async () => {
    await render(developerSettings.fields.find((field) => field.id === 'developerLogNamespaces')!);
    expect(modifiedIndicators()).toHaveLength(0);
    const checkbox = host.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
    const initial = checkbox.checked;
    await act(async () => {
      checkbox.click();
      await Promise.resolve();
    });
    expect(checkbox.checked).toBe(!initial);
    expect(modifiedIndicators()).toHaveLength(1);
    await act(async () => {
      checkbox.click();
      await Promise.resolve();
    });
    expect(modifiedIndicators()).toHaveLength(0);
  });

  it('lists shortcuts under category headers and says so when a search matches nothing', async () => {
    host.style.cssText = 'display:flex;flex-direction:column;height:600px;width:640px;';
    await render(hotkeysSettings.fields[0]!);
    await expect.poll(() => host.textContent).toContain('Open Command Palette');

    const list = host.querySelector<HTMLElement>('[role="list"][aria-label="hotkeys.bindings"]')!;

    expect(list).not.toBeNull();
    expect(list.querySelectorAll('[role="listitem"]').length).toBeGreaterThan(0);
    // Shortcut editors are separated by hairlines, one fewer than the rows in a section.
    expect(list.querySelectorAll('[aria-hidden="true"]').length).toBeGreaterThan(0);
    expect(host.querySelector('[data-list-pinned-header]')?.textContent).toContain('hotkeys.categories.app');

    const search = host.querySelector<HTMLInputElement>('input[aria-label="hotkeys.searchPlaceholder"]')!;

    await act(() => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
      setter.call(search, 'no shortcut is called this');
      search.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect.poll(() => host.textContent).toContain('hotkeys.noMatches');
    expect(host.querySelector('[role="list"][aria-label="hotkeys.bindings"]')).toBeNull();
  });

  it('marks a disabled shortcut but does not mark an explicit binding equal to its declared default', async () => {
    const errorSpy = vi.spyOn(console, 'error');
    await patchWorkbenchPreferences({ customHotkeys: { 'app.openCommandPalette': ['mod+k'] } });
    await render(hotkeysSettings.fields[0]!);
    await expect.poll(() => host.textContent).toContain('Open Command Palette');
    expect(modifiedIndicators()).toHaveLength(0);
    await act(() => patchWorkbenchPreferences({ customHotkeys: { 'app.openCommandPalette': [] } }));
    await expect.poll(() => modifiedIndicators().length).toBe(1);
    expect(modifiedIndicators()[0]!.getAttribute('aria-label')).toBe('Open Command Palette differs from its default');
    await act(() => patchWorkbenchPreferences({ customHotkeys: {} }));
    expect(modifiedIndicators()).toHaveLength(0);
    const snapshotWarnings = errorSpy.mock.calls.filter(([message]) =>
      String(message).includes('getSnapshot should be cached')
    );
    errorSpy.mockRestore();
    expect(snapshotWarnings).toEqual([]);
  });
});

describe('workspace settings', () => {
  it('opens the intermediates manager through the surface hosting it', async () => {
    const onReveal = vi.fn();
    await render(workspaceSettings.fields[0]!, onReveal);
    const manage = [...host.querySelectorAll('button')].find(
      (button) => button.textContent === i18n.t('settings.catalog.manageIntermediates')
    );
    await act(() => manage!.click());
    expect(onReveal).toHaveBeenCalledWith('intermediates', 'intermediatesManager');
  });
});
