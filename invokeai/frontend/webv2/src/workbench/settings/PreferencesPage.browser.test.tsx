import type * as Languages from '@platform/i18n/languages';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { SettingFieldProps } from '@platform/ui/settings/contracts';
import type * as projectsApi from '@workbench/projects/api';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  RouterProvider,
} from '@tanstack/react-router';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { PreferencesPage } from './PreferencesPage';
import { DEFAULT_PREFERENCES, getWorkbenchPreferences, patchWorkbenchPreferences } from './store';

vi.mock('@platform/i18n/languages', async (importOriginal) => {
  const original = await importOriginal<typeof Languages>();
  const catalogs = { en: { translated: 10, ready: true }, fr: { translated: 2 } };
  return {
    ...original,
    getWorkbenchLanguageAvailability: (language: Languages.WorkbenchLanguage) =>
      original.getWorkbenchLanguageAvailability(language, catalogs),
    getEffectiveWorkbenchLanguage: (language: Languages.WorkbenchLanguage) =>
      original.getEffectiveWorkbenchLanguage(language, catalogs),
  };
});

vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));

vi.mock('./CustomSettingsEditors', () => ({
  default: ({ field }: SettingFieldProps) => <div data-custom-editor={field.id}>Custom settings editor</div>,
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

const validateSearch = (search: Record<string, unknown>): { setting?: string } => ({
  setting: typeof search.setting === 'string' && search.setting.length > 0 ? search.setting : undefined,
});

/** Like the Launchpad, the page stays mounted while another section's route is showing. */
const createTestRouter = (path: string) => {
  const rootRoute = createRootRoute({
    component: () => (
      <>
        <PreferencesPage />
        <Outlet />
      </>
    ),
  });
  return createRouter({
    history: createMemoryHistory({ initialEntries: [path] }),
    routeTree: rootRoute.addChildren([
      createRoute({ getParentRoute: () => rootRoute, path: 'preferences', validateSearch }),
      createRoute({ getParentRoute: () => rootRoute, path: 'preferences/$section', validateSearch }),
      createRoute({ getParentRoute: () => rootRoute, path: 'fonts' }),
    ]),
  });
};

const render = async (path: string) => {
  const router = createTestRouter(path);
  await act(async () => {
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <RouterProvider router={router} />
        </I18nextProvider>
      </ChakraProvider>
    );
    await router.load();
  });
  return router;
};

const search = () => page.getByRole('textbox', { name: i18n.t('settingsDialog.search'), exact: true });
const sectionItem = (name: string) => page.getByRole('button', { name, exact: true });
const detailHeading = (name: string) => page.getByRole('heading', { level: 2, name, exact: true });

beforeEach(async () => {
  await page.viewport(1280, 900);
  accountLifecycle.activate('preferences-test');
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

describe('Launchpad preferences page', () => {
  it('lists every section, per-project ones as defaults, and follows the route', async () => {
    const router = await render('/preferences/behavior');

    await expect.element(detailHeading('Behavior')).toBeVisible();
    await expect.element(sectionItem('Behavior')).toHaveAttribute('aria-current', 'true');
    for (const name of [
      'Appearance',
      'Keyboard shortcuts',
      'Generate',
      'Canvas',
      'Gallery',
      'Image Map',
      'Preview',
      'Workflow',
      'Queue',
      'Developer',
    ]) {
      await expect.element(sectionItem(name)).toBeInTheDocument();
    }
    // Server needs admin rights; there is no Project group any more.
    await expect.element(sectionItem('Server')).not.toBeInTheDocument();
    expect(host.textContent).not.toContain('Project');

    await act(() => sectionItem('Canvas').click());
    await expect.poll(() => router.state.location.pathname).toBe('/preferences/canvas');
    await expect.element(page.getByText(i18n.t('settingsDialog.defaultsNote'), { exact: true })).toBeVisible();
    const grid = page.getByText(i18n.t('widgets.canvas.settings.grid'), { exact: true });
    await expect.element(grid).toBeVisible();
    await act(() => grid.click());
    await expect.poll(() => getWorkbenchPreferences().scopedDefaults.canvas).toEqual({ showGrid: true });

    await act(() => sectionItem('Appearance').click());
    await expect.poll(() => router.state.location.pathname).toBe('/preferences/appearance');
    await expect.element(detailHeading('Appearance')).toBeVisible();
    await expect
      .element(page.getByText(i18n.t('settingsDialog.defaultsNote'), { exact: true }))
      .not.toBeInTheDocument();
  });

  it('narrows sections while searching and reveals a result in its section', async () => {
    const router = await render('/preferences/appearance');

    await act(() => search().fill('numeric attention'));
    await expect.element(detailHeading(i18n.t('settingsDialog.results'))).toBeVisible();
    await expect.element(sectionItem('Appearance')).not.toBeInTheDocument();
    // One match overall, in Behavior.
    await expect
      .element(page.getByRole('listitem').filter({ hasText: i18n.t('settingsDialog.allResults') }))
      .toHaveTextContent(`${i18n.t('settingsDialog.allResults')}1`);
    await expect.element(page.getByRole('listitem').filter({ hasText: 'Behavior' })).toHaveTextContent('Behavior1');
    await expect.element(page.getByText('Prefer numeric attention style', { exact: true })).toBeVisible();

    await act(() => page.getByRole('button', { name: i18n.t('settingsDialog.showInSection'), exact: true }).click());
    await expect.element(search()).toHaveValue('');
    await expect.element(detailHeading('Behavior')).toBeVisible();
    await expect
      .element(page.getByRole('checkbox', { name: 'Prefer numeric attention style', exact: true }))
      .toHaveFocus();
    // The reveal request is spent once the entry has focus.
    await expect.poll(() => router.state.location.href).toBe('/preferences/behavior');

    await act(() => search().fill('no such setting'));
    await act(() =>
      page
        .getByRole('button', { name: i18n.t('settingsDialog.clearSearch'), exact: true })
        .last()
        .click()
    );
    await expect.element(search()).toHaveValue('');
    await expect.element(detailHeading('Behavior')).toBeVisible();
  });

  it('edits a per-project setting found by search as the default, and resets it to the built-in', async () => {
    await render('/preferences/appearance');

    await act(() => search().fill('NSFW'));
    const toggle = page.getByRole('checkbox', { name: 'NSFW checker', exact: true });
    await expect.element(toggle).toBeEnabled();
    await expect.element(toggle).not.toBeChecked();
    await act(() => page.getByText('NSFW checker', { exact: true }).click());
    await expect.poll(() => getWorkbenchPreferences().scopedDefaults.generate).toEqual({ useNsfwChecker: true });

    await act(() => page.getByRole('button', { name: i18n.t('settingsDialog.showInSection'), exact: true }).click());
    await expect.element(detailHeading('Generate')).toBeVisible();
    await expect.element(page.getByText(i18n.t('settingsDialog.defaultsNote'), { exact: true })).toBeVisible();
    await act(() =>
      page
        .getByRole('button', {
          name: i18n.t('settingsDialog.layer.changedNamed', { setting: 'NSFW checker' }),
          exact: true,
        })
        .click()
    );
    await act(() => page.getByRole('menuitem', { name: i18n.t('settingsDialog.layer.resetToBuiltIn') }).click());
    await expect.poll(() => getWorkbenchPreferences().scopedDefaults).toEqual({});
    await expect.element(toggle).not.toBeChecked();
  });
});

describe('Launchpad preferences page in a single pane', () => {
  const back = () => page.getByRole('button', { name: i18n.t('settingsDialog.backToList'), exact: true });

  beforeEach(() => {
    // Below the 50rem both panes need.
    host.style.width = '640px';
  });

  it('moves focus between a section and its row, and keeps browser history on the same panes', async () => {
    const router = await render('/preferences/appearance');

    await back().click();
    await expect.poll(() => router.state.location.pathname).toBe('/preferences');
    await expect.element(sectionItem('Appearance')).toHaveFocus();

    await act(() => sectionItem('Behavior').click());
    await expect.poll(() => router.state.location.pathname).toBe('/preferences/behavior');
    await expect.element(detailHeading('Behavior')).toBeVisible();
    await expect.element(back()).toHaveFocus();

    // The browser's Back returns to the list Back left, not to the Appearance settings before it.
    await act(async () => {
      router.history.back();
      await new Promise((resolve) => {
        setTimeout(resolve, 50);
      });
    });
    await expect.poll(() => router.state.location.pathname).toBe('/preferences');
    await expect.element(sectionItem('Behavior')).toBeVisible();
    await expect.element(detailHeading('Behavior')).not.toBeInTheDocument();
  });
});
