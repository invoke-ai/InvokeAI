/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { SettingFieldProps } from '@platform/ui/settings/contracts';
import type * as projectsApi from '@workbench/projects/api';
import type { WidgetRuntimeApi, WidgetViewProps } from '@workbench/widgetContracts';
import type * as workbenchContext from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { queueWidgetManifest } from '@workbench/widgets/queue/manifest';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { appearanceSettings } from './applicationContributions';
import { SettingsDialogHost } from './SettingsDialogHost';
import {
  closeWorkbenchSettings,
  getSettingsSectionScroll,
  openWorkbenchSettings,
  settingsDialogStore,
} from './settingsDialogStore';
import { DEFAULT_PREFERENCES, getWorkbenchPreferences, patchWorkbenchPreferences } from './store';
import { WidgetSettingsButton } from './WidgetSettingsButton';

vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));

vi.mock('./CustomSettingsEditors', () => ({
  default: ({ field }: SettingFieldProps) => <div data-custom-editor={field.id}>Custom settings editor</div>,
}));

const testProject = {
  id: 'settings-test-project',
  settings: { antialiasProgressImages: true, showProgressImagesInViewer: true, useCpuNoise: false },
  widgetInstances: { 'settings-test-widget': { id: 'settings-test-widget', typeId: 'queue' } },
};

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof workbenchContext>()),
  useOptionalWorkbenchSelector: (selector: (snapshot: never) => unknown) =>
    selector({ activeProject: testProject } as never),
  useActiveProjectId: () => 'settings-test-project',
  useWorkbenchQueries: () => ({
    isActiveProject: (projectId: string) => projectId === 'settings-test-project',
    getProject: () => ({ widgetInstances: { 'settings-test-widget': { typeId: 'queue' } } }),
  }),
}));

const translations = await fetch('/locales/en.json').then((response) => response.json());
const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  fallbackLng: 'en',
  resources: { en: { translation: translations } },
  interpolation: { escapeValue: false },
});

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const quickWidgetProps: WidgetViewProps = {
  instance: { id: 'settings-test-widget', typeId: 'queue', createdAt: '2026-01-01T00:00:00.000Z' },
  manifest: {
    ...queueWidgetManifest,
    settings: { ...appearanceSettings, quick: ['showFocusRegionHighlight'] },
  },
  region: 'right',
  runtime: {} as WidgetRuntimeApi,
};

const render = async (quick = false) => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <button type="button" onClick={() => openWorkbenchSettings('behavior')}>
            Open settings
          </button>
          {quick ? <WidgetSettingsButton {...quickWidgetProps} /> : null}
          <SettingsDialogHost />
        </I18nextProvider>
      </ChakraProvider>
    )
  );
};

const dialogContent = () => document.querySelector<HTMLElement>('[data-scope="dialog"][data-part="content"]');
const search = () => page.getByRole('textbox', { name: i18n.t('settingsDialog.search'), exact: true });
const closeButton = () => page.getByRole('button', { name: i18n.t('common.close'), exact: true });
const waitForOpen = async () => {
  await expect.element(search()).toBeVisible();
  await expect.poll(() => dialogContent()?.getAnimations().length).toBe(0);
};
const open = async () => {
  await act(() => page.getByRole('button', { name: 'Open settings', exact: true }).click());
  await waitForOpen();
};

beforeEach(async () => {
  await page.viewport(1280, 900);
  delete document.documentElement.dataset.reduceMotion;
  accountLifecycle.activate('settings-test');
  settingsDialogStore.patchSnapshot({
    isOpen: false,
    sectionId: 'appearance',
    query: '',
    searchSection: null,
    target: undefined,
    entryId: undefined,
    returnFocus: null,
  });
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  delete document.documentElement.dataset.reduceMotion;
  closeWorkbenchSettings();
  accountLifecycle.invalidate();
});

describe('settings dialog', () => {
  it('edits a matching preference and reveals it in its section without losing its value', async () => {
    await render();
    await open();
    await act(() => search().fill('numeric attention'));
    await expect.element(page.getByText('Prefer numeric attention style', { exact: true })).toBeVisible();
    await expect.element(search()).toHaveFocus();
    const modifiedIndicator = page.getByRole('img', {
      name: 'Prefer numeric attention style: changed from default',
      exact: true,
    });
    await expect.element(modifiedIndicator).not.toBeInTheDocument();
    await act(() => page.getByText('Prefer numeric attention style', { exact: true }).click());
    await expect.poll(() => getWorkbenchPreferences().preferNumericAttentionStyle).toBe(true);
    await expect.element(modifiedIndicator).toBeVisible();
    await act(() => page.getByRole('button', { name: i18n.t('settingsDialog.showInSection'), exact: true }).click());
    await expect.element(search()).toHaveValue('');
    await expect.element(page.getByRole('dialog', { name: 'Settings: Behavior', exact: true })).toBeVisible();
    await expect
      .element(page.getByRole('checkbox', { name: 'Prefer numeric attention style', exact: true }))
      .toBeChecked();
    await expect
      .element(page.getByRole('button', { name: 'Behavior', exact: true }))
      .toHaveAttribute('aria-current', 'page');
    await expect.element(modifiedIndicator).toBeVisible();
    await act(() => page.getByText('Prefer numeric attention style', { exact: true }).click());
    await expect.poll(() => getWorkbenchPreferences().preferNumericAttentionStyle).toBe(false);
    await expect.element(modifiedIndicator).not.toBeInTheDocument();
  });

  it('lists project settings, and widget settings only for widgets in the open project', async () => {
    await render();
    await open();
    const nav = page.getByRole('navigation', { name: i18n.t('settings.title'), exact: true });
    await expect.element(nav.getByRole('button', { name: 'Project', exact: true })).toBeVisible();
    await expect.element(nav.getByRole('button', { name: 'Queue', exact: true })).toBeVisible();
    // Canvas settings are all per widget and this project has no canvas; Preview keeps its project setting.
    await expect.element(nav.getByRole('button', { name: 'Canvas', exact: true })).not.toBeInTheDocument();
    await act(() => nav.getByRole('button', { name: 'Preview', exact: true }).click());
    await expect.element(page.getByText('Antialias progress images', { exact: true })).toBeVisible();
    await expect
      .element(page.getByText(i18n.t('settingsDialog.instanceUnavailable'), { exact: true }))
      .not.toBeInTheDocument();
  });

  it('switches prepared sections without loading placeholders or mounting unselected custom editors', async () => {
    await render();
    const mountedCustomEditors: string[] = [];
    const loadingPlaceholders: string[] = [];
    let opened = false;
    const observer = new MutationObserver((records) => {
      for (const record of records) {
        for (const node of record.addedNodes) {
          if (!(node instanceof Element)) {
            continue;
          }
          const addedElements = [node, ...node.querySelectorAll('[role="status"], [data-custom-editor]')];
          for (const element of addedElements) {
            const editorId = element.getAttribute('data-custom-editor');
            if (editorId) {
              mountedCustomEditors.push(editorId);
            }
            if (
              opened &&
              element.getAttribute('role') === 'status' &&
              element.textContent?.includes(i18n.t('common.loading'))
            ) {
              loadingPlaceholders.push(element.textContent);
            }
          }
        }
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    try {
      await open();
      expect(mountedCustomEditors).toEqual([]);
      opened = true;
      for (const section of ['Workflow', 'Queue', 'Appearance', 'Behavior']) {
        await act(() => page.getByRole('button', { name: section, exact: true }).click());
        await expect.element(page.getByRole('dialog', { name: `Settings: ${section}`, exact: true })).toBeVisible();
        expect(dialogContent()?.querySelector('[data-custom-editor]')?.getAttribute('data-custom-editor') ?? null).toBe(
          section === 'Appearance' ? 'themeId' : null
        );
      }
      expect(loadingPlaceholders).toEqual([]);
      expect(mountedCustomEditors).toEqual(['themeId']);
    } finally {
      observer.disconnect();
    }
  });

  it('reveals a search result through the section selector in a narrow window', async () => {
    await page.viewport(360, 780);
    await render();
    await open();
    await act(() => search().fill('numeric attention'));
    await expect.element(page.getByText('Prefer numeric attention style', { exact: true })).toBeVisible();
    await act(() => page.getByRole('button', { name: i18n.t('settingsDialog.showInSection'), exact: true }).click());
    await expect
      .element(page.getByRole('combobox', { name: i18n.t('settingsDialog.section'), exact: true }))
      .toHaveValue('behavior');
    await expect.element(page.getByRole('dialog', { name: 'Settings: Behavior', exact: true })).toBeVisible();
    await expect
      .element(page.getByRole('checkbox', { name: 'Prefer numeric attention style', exact: true }))
      .toHaveFocus();
  });

  it('restores section scroll after searching, switching sections, and reopening', async () => {
    await page.viewport(1280, 480);
    await render();
    await open();
    const body = () => {
      const label = page.getByText('Prefer numeric attention style', { exact: true }).element();
      let element = label.parentElement;
      while (element && getComputedStyle(element).overflowY !== 'auto') {
        element = element.parentElement;
      }
      return element!;
    };
    await act(() => {
      body().scrollTop = 100;
    });
    await expect.poll(() => getSettingsSectionScroll('behavior')).toBe(100);
    await act(() => search().fill('numeric attention'));
    await act(() => search().fill(''));
    await expect.poll(() => body().scrollTop).toBe(100);
    await act(() => page.getByRole('button', { name: 'Queue', exact: true }).click());
    await act(() => page.getByRole('button', { name: 'Behavior', exact: true }).click());
    await expect.poll(() => body().scrollTop).toBe(100);
    await act(() => closeButton().click());
    await expect.poll(dialogContent).toBeNull();
    await open();
    await expect.poll(() => body().scrollTop).toBe(100);
  });

  it('keeps closing content present for animation, supports reopening, and restores trigger focus', async () => {
    await render();
    await open();
    const content = dialogContent()!;
    await act(() => closeWorkbenchSettings());
    expect(content.isConnected).toBe(true);
    expect(content.dataset.state).toBe('closed');
    expect(getComputedStyle(content).animationName).not.toBe('none');
    await act(() => openWorkbenchSettings('behavior', host.querySelector('button')!));
    await waitForOpen();
    expect(dialogContent()).toBe(content);
    await act(() => closeButton().click());
    await expect.poll(dialogContent).toBeNull();
    await expect.element(page.getByRole('button', { name: 'Open settings', exact: true })).toHaveFocus();
  });

  it('dismisses with Escape without retained content under reduced motion', async () => {
    document.documentElement.dataset.reduceMotion = 'true';
    await render();
    await open();
    const content = dialogContent()!;
    expect(parseFloat(getComputedStyle(content).animationDuration)).toBeLessThanOrEqual(0.001);
    await act(() => userEvent.keyboard('{Escape}'));
    await expect.poll(dialogContent).toBeNull();
    await expect.element(page.getByRole('button', { name: 'Open settings', exact: true })).toHaveFocus();
  });

  it('returns focus to the widget gear when quick settings close with Escape', async () => {
    await render(true);
    const gear = page.getByRole('button', {
      name: i18n.t('widgets.settingsLabel', { label: 'Appearance' }),
      exact: true,
    });
    await act(() => gear.click());
    await expect.element(page.getByText('Highlight focused regions', { exact: true })).toBeVisible();
    await act(() => userEvent.keyboard('{Escape}'));
    await expect.poll(() => document.querySelector('[data-scope="popover"][data-part="content"]')).toBeNull();
    await expect.element(gear).toHaveFocus();
  });

  it('hands quick settings to the full section and restores the widget gear', async () => {
    await render(true);
    const gear = page.getByRole('button', {
      name: i18n.t('widgets.settingsLabel', { label: 'Appearance' }),
      exact: true,
    });
    await act(() => gear.click());
    const quickLabel = page.getByText('Highlight focused regions', { exact: true });
    await expect.element(quickLabel).toBeVisible();
    await act(() => quickLabel.click());
    await expect.poll(() => getWorkbenchPreferences().showFocusRegionHighlight).toBe(false);
    await page
      .getByRole('button', { name: i18n.t('settingsDialog.allWidgetSettings', { widget: 'Appearance' }), exact: true })
      .click();
    await waitForOpen();
    await expect.element(page.getByRole('dialog', { name: 'Settings: Appearance', exact: true })).toBeVisible();
    expect(settingsDialogStore.getSnapshot().target).toEqual({
      projectId: 'settings-test-project',
      instanceId: 'settings-test-widget',
    });
    await expect.poll(() => dialogContent()?.contains(document.activeElement)).toBe(true);
    await expect
      .element(page.getByRole('checkbox', { name: 'Highlight focused regions', exact: true }))
      .not.toBeChecked();
    await expect.poll(() => document.querySelector('[data-scope="popover"][data-part="content"]')).toBeNull();
    await act(() => closeButton().click());
    await expect.poll(dialogContent).toBeNull();
    await expect.element(gear).toHaveFocus();
  });
});
