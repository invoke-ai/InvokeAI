/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { ProjectSummaryDTO } from '@workbench/projects/api';
import type * as projectsApi from '@workbench/projects/api';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  RouterProvider,
} from '@tanstack/react-router';
import { system } from '@theme/system';
import { seedProjectLibrary } from '@workbench/projects/library';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ClearSavedDataSettings } from './ClearSavedDataSettings';

const clear = vi.hoisted(() => ({ clearWorkspaceData: vi.fn(), listProjects: vi.fn() }));

vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  listProjects: clear.listProjects,
}));
vi.mock('@workbench/projects/covers', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  loadProjectCovers: () => Promise.resolve(),
}));

// Clearing reloads the page on success, so the test stops at the request: it never settles.
vi.mock('./clearWorkspaceData', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  clearWorkspaceData: clear.clearWorkspaceData,
}));

const translations = await fetch('/locales/en.json').then((response) => response.json());
const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: translations } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

const summary = (id: string): ProjectSummaryDTO =>
  ({
    created_at: '2026-10-01T00:00:00Z',
    minimum_canvas_schema_version: 1,
    name: id,
    project_id: id,
    revision: 1,
    updated_at: '2026-10-01T00:00:00Z',
  }) as ProjectSummaryDTO;

const render = async () => {
  const rootRoute = createRootRoute({
    component: () => (
      <>
        <ClearSavedDataSettings description="Delete everything" label="Clear saved data" />
        <Outlet />
      </>
    ),
  });
  const router = createRouter({
    history: createMemoryHistory({ initialEntries: ['/preferences'] }),
    routeTree: rootRoute.addChildren([
      createRoute({ getParentRoute: () => rootRoute, path: 'preferences' }),
      createRoute({ getParentRoute: () => rootRoute, path: 'projects' }),
    ]),
  });
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

const dialog = () => page.getByRole('alertdialog');
const dialogText = () => document.querySelector('[role="alertdialog"]')?.textContent ?? '';
const confirmButton = () => dialog().getByRole('button', { name: i18n.t('settings.clearData.confirm'), exact: true });

beforeEach(() => {
  accountLifecycle.activate('clear-saved-data-test');
  clear.clearWorkspaceData.mockReset().mockReturnValue(new Promise<never>(() => {}));
  clear.listProjects.mockReset();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

const word = () =>
  dialog().getByRole('textbox', {
    name: i18n.t('settings.clearData.typeToConfirm', { word: i18n.t('settings.clearData.confirmWord') }),
  });
const openDialog = () =>
  userEvent.click(page.getByRole('button', { name: i18n.t('settings.clearData.action'), exact: true }));

describe('clear saved data', () => {
  it('counts the projects afresh, names what stays, and deletes only once the typed word matches', async () => {
    // The library on hand is stale: another tab has made a third project since.
    seedProjectLibrary([summary('a'), summary('b')], captureAccountScope());
    clear.listProjects.mockResolvedValue([summary('a'), summary('b'), summary('c')]);
    await render();

    await openDialog();

    await expect.poll(dialogText).toContain('all 3 of your projects');
    expect(dialogText()).toContain(i18n.t('settings.clearData.mediaKept'));
    // The word guards the confirm, so the field takes focus; Enter confirms only once it matches.
    await expect.element(word()).toHaveFocus();
    await expect.element(confirmButton()).toBeDisabled();
    await userEvent.keyboard('delete{Enter}');
    await expect.element(confirmButton()).toBeDisabled();
    expect(clear.clearWorkspaceData).not.toHaveBeenCalled();

    await userEvent.fill(word(), i18n.t('settings.clearData.confirmWord'));
    await userEvent.keyboard('{Enter}');
    expect(clear.clearWorkspaceData).toHaveBeenCalledOnce();
    // Nowhere else to go while it runs.
    await expect
      .element(dialog().getByRole('button', { name: i18n.t('settings.clearData.openProjects'), exact: true }))
      .toBeDisabled();
  });

  it('says every saved project goes while the count is unknown, and says so plainly when there are none', async () => {
    clear.listProjects.mockReturnValue(new Promise<never>(() => {}));
    await render();
    await openDialog();
    await expect.poll(dialogText).toContain(i18n.t('settings.clearData.projectCountUnknown'));
    await userEvent.click(dialog().getByRole('button', { name: i18n.t('common.cancel'), exact: true }));

    await act(() => root.unmount());
    root = createRoot(host);
    // A new account lifetime: the listing that never answered belonged to the last one.
    accountLifecycle.invalidate();
    accountLifecycle.activate('clear-saved-data-test-2');
    clear.listProjects.mockResolvedValue([]);
    await render();
    await openDialog();
    await expect.poll(dialogText).toContain(i18n.t('settings.clearData.bodyInstallNoProjects'));
  });

  it('starts every opening without the word, and sends the viewer to Projects to export first', async () => {
    clear.listProjects.mockResolvedValue([summary('a')]);
    const router = await render();

    await openDialog();
    await expect.poll(dialogText).toContain('deletes your 1 project and');
    await userEvent.fill(word(), i18n.t('settings.clearData.confirmWord'));
    await userEvent.click(dialog().getByRole('button', { name: i18n.t('common.cancel'), exact: true }));
    await expect.element(dialog()).not.toBeInTheDocument();

    await openDialog();
    await expect.element(word()).toHaveValue('');

    await userEvent.click(
      dialog().getByRole('button', { name: i18n.t('settings.clearData.openProjects'), exact: true })
    );
    await expect.poll(() => router.state.location.pathname).toBe('/projects');
    expect(clear.clearWorkspaceData).not.toHaveBeenCalled();
  });
});
