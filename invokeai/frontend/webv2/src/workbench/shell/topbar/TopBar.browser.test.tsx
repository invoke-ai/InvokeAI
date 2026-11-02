import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { behaviorSettings } from '@workbench/settings/applicationContributions';
import { ApplicationSettingField } from '@workbench/settings/ApplicationSettingField';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { TopBar } from './TopBar';

vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));
vi.mock('@workbench/settings/SettingsDialogHost', () => ({ SettingsDialogHost: () => null }));
vi.mock('@workbench/shell/WidgetsDialog', () => ({ WidgetsDialog: () => null }));
vi.mock('@workbench/palette/PaletteButton', () => ({ PaletteButton: () => <button>Palette</button> }));
vi.mock('./LayoutPresetManagerDialog', () => ({ LayoutPresetManagerDialog: () => null }));
vi.mock('./LayoutPresetAdminDialogs', () => ({ LayoutPresetAdminDialogs: () => null }));
vi.mock('./TopbarProgressRail', () => ({ TopbarProgressRail: () => null }));
vi.mock('./AppMenu', () => ({ AppMenu: () => <button>Menu</button> }));
vi.mock('./ProjectSwitcher', () => ({ ProjectSwitcher: () => <button>Project</button> }));
vi.mock('./LayoutPresetStrip', () => ({ LayoutPresetStrip: () => <span>Presets</span> }));
vi.mock('./QueueCluster', () => ({ QueueCluster: () => <button>Queue</button> }));
vi.mock('./InvocationCluster', () => ({
  InvocationCluster: () => (
    <div>
      <input aria-label="Iterations" defaultValue="1" />
      <button>Invoke</button>
    </div>
  ),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  accountLifecycle.invalidate();
});

it.each([1024, 1280])(
  'positions Invoke using its Behavior setting and keeps presets centered at %ipx',
  async (width) => {
    await page.viewport(width, 300);
    accountLifecycle.activate('invoke-placement-test');
    await patchWorkbenchPreferences({ showInvokeButtonOnLeft: false });
    const i18n = createInstance();
    await i18n.use(initReactI18next).init({ lng: 'en', resources: { en: { translation: {} } } });
    const field = behaviorSettings.fields.find(({ id }) => id === 'showInvokeButtonOnLeft')!;
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <TopBar />
            <ApplicationSettingField field={field} surface="dialog" />
          </I18nextProvider>
        </ChakraProvider>
      );
    });
    const group = () => host.querySelector<HTMLElement>('[data-invocation-side]')!;
    const project = page.getByRole('button', { name: 'Project', exact: true }).element();
    const menu = page.getByRole('button', { name: 'Menu', exact: true }).element();
    const queue = page.getByRole('button', { name: 'Queue', exact: true }).element();
    const header = host.querySelector('header')!;
    const presets = page.getByText('Presets', { exact: true }).element();
    const expectCenteredPresets = () => {
      const headerBounds = header.getBoundingClientRect();
      const presetBounds = presets.getBoundingClientRect();
      expect(presetBounds.left + presetBounds.width / 2).toBeCloseTo(headerBounds.left + headerBounds.width / 2, 1);
    };
    const setting = page.getByRole('checkbox', { name: 'Show Invoke button on the left', exact: true });
    const settingLabel = page.getByText('Show Invoke button on the left', { exact: true });
    await expect.element(setting).not.toBeChecked();
    expect(group().dataset.invocationSide).toBe('right');
    expect(project.getBoundingClientRect().right).toBeLessThanOrEqual(group().getBoundingClientRect().left);
    expectCenteredPresets();

    await act(() => userEvent.click(settingLabel));
    await expect.element(setting).toBeChecked();
    expect(group().dataset.invocationSide).toBe('left');
    expect(menu.getBoundingClientRect().right).toBeLessThanOrEqual(group().getBoundingClientRect().left);
    expect(group().getBoundingClientRect().right).toBeLessThanOrEqual(project.getBoundingClientRect().left);
    expect(group().getBoundingClientRect().right).toBeLessThanOrEqual(queue.getBoundingClientRect().left);
    expectCenteredPresets();

    await act(() => userEvent.click(settingLabel));
    await expect.element(setting).not.toBeChecked();
    expect(group().dataset.invocationSide).toBe('right');
    expect(project.getBoundingClientRect().right).toBeLessThanOrEqual(group().getBoundingClientRect().left);
    expect(group().getBoundingClientRect().right).toBeLessThanOrEqual(queue.getBoundingClientRect().left);
    expectCenteredPresets();
  }
);
import type * as projectsApi from '@workbench/projects/api';
