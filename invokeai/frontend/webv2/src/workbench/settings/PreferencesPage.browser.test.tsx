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
import { page, userEvent } from 'vitest/browser';

import { PreferencesPage } from './PreferencesPage';
import { DEFAULT_PREFERENCES, getWorkbenchPreferences, patchWorkbenchPreferences } from './store';

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
  it('lists only settings that need no open project and follows the route', async () => {
    const router = await render('/preferences/behavior');

    await expect.element(detailHeading('Behavior')).toBeVisible();
    await expect.element(sectionItem('Behavior')).toHaveAttribute('aria-current', 'true');
    for (const name of ['Appearance', 'Keyboard shortcuts', 'Gallery', 'Workflow', 'Queue', 'Developer']) {
      await expect.element(sectionItem(name)).toBeInTheDocument();
    }
    // Project, and widget sections with no preference entries, need a project; Server needs admin rights.
    for (const name of ['Project', 'Preview', 'Canvas', 'Server']) {
      await expect.element(sectionItem(name)).not.toBeInTheDocument();
    }

    await act(() => sectionItem('Appearance').click());
    await expect.poll(() => router.state.location.pathname).toBe('/preferences/appearance');
    await expect.element(detailHeading('Appearance')).toBeVisible();
  });

  it('edits a preference through the shared settings store', async () => {
    await render('/preferences/appearance');

    await act(() => page.getByText('Reduce motion', { exact: true }).click());
    await expect.poll(() => getWorkbenchPreferences().reduceMotion).toBe(true);
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

  it('replaces a search left behind when another page asks for a setting', async () => {
    const router = await render('/preferences/appearance');
    await act(() => search().fill('motion'));
    await expect.element(detailHeading(i18n.t('settingsDialog.results'))).toBeVisible();

    // The palette and other Launchpad pages link straight to a setting while this page stays mounted.
    await act(() => router.navigate({ to: '/fonts' }));
    await act(() =>
      router.navigate({
        params: { section: 'behavior' },
        search: { setting: 'preferNumericAttentionStyle' },
        to: '/preferences/$section',
      })
    );
    await expect.element(search()).toHaveValue('');
    await expect.element(detailHeading('Behavior')).toBeVisible();
    await expect
      .element(page.getByRole('checkbox', { name: 'Prefer numeric attention style', exact: true }))
      .toHaveFocus();
    await expect.poll(() => router.state.location.href).toBe('/preferences/behavior');
  });

  it('keeps a search when the rail link returns to the page', async () => {
    const router = await render('/preferences/appearance');
    await act(() => search().fill('motion'));
    await act(() => router.navigate({ to: '/fonts' }));
    await act(() => router.navigate({ to: '/preferences' }));
    await expect.element(search()).toHaveValue('motion');
    await expect.element(detailHeading(i18n.t('settingsDialog.results'))).toBeVisible();
  });

  it('jumps to search on `/` unless a field is being typed in', async () => {
    await render('/preferences/appearance');

    await act(() => sectionItem('Behavior').click());
    await expect.element(sectionItem('Behavior')).toHaveFocus();
    await userEvent.keyboard('/');
    await expect.element(search()).toHaveFocus();
    await userEvent.keyboard('/');
    await expect.element(search()).toHaveValue('/');
  });

  it('returns to the last section from the bare route and corrects unknown sections', async () => {
    const router = await render('/preferences/hotkeys');

    await expect.element(detailHeading('Keyboard shortcuts')).toBeVisible();
    await act(() => router.navigate({ to: '/fonts' }));
    await act(() => router.navigate({ to: '/preferences' }));
    await expect.element(detailHeading('Keyboard shortcuts')).toBeVisible();

    await act(() => router.navigate({ params: { section: 'project' }, to: '/preferences/$section' }));
    await expect.poll(() => router.state.location.pathname).toBe('/preferences/hotkeys');
  });
});

describe('Launchpad preferences page in a single pane', () => {
  const back = () => page.getByRole('button', { name: i18n.t('settingsDialog.backToList'), exact: true });

  beforeEach(() => {
    // Below the 50rem both panes need.
    host.style.width = '640px';
  });

  it('opens on the section list from the bare route and on a section from its route', async () => {
    await render('/preferences');

    await expect.element(sectionItem('Behavior')).toBeVisible();
    await expect.element(detailHeading('Appearance')).not.toBeInTheDocument();
    await act(() => root.unmount());
    root = createRoot(host);

    await render('/preferences/appearance');

    await expect.element(detailHeading('Appearance')).toBeVisible();
    await expect.element(back()).toBeVisible();
    await expect.element(sectionItem('Behavior')).not.toBeInTheDocument();
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

  it('opens search results from the list and returns to it', async () => {
    await render('/preferences');

    await act(() => search().fill('numeric attention'));
    await act(() => page.getByRole('button', { name: new RegExp(`^${i18n.t('settingsDialog.allResults')}`) }).click());

    await expect.element(detailHeading(i18n.t('settingsDialog.results'))).toBeVisible();
    await expect.element(page.getByText('Prefer numeric attention style', { exact: true })).toBeVisible();
    await back().click();
    await expect.element(search()).toHaveValue('numeric attention');
    await expect.element(detailHeading(i18n.t('settingsDialog.results'))).not.toBeInTheDocument();
  });

  it('leaves `/` alone while the search field is out of view', async () => {
    await render('/preferences/appearance');
    await back().click();
    await act(() => sectionItem('Behavior').click());
    await expect.element(back()).toHaveFocus();

    const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: '/' });
    back().element().dispatchEvent(event);

    expect(event.defaultPrevented).toBe(false);
    await expect.element(back()).toHaveFocus();
  });
});
