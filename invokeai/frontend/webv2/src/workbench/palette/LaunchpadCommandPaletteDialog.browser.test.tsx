import type * as libraryModule from '@workbench/projects/library';
import type { ProjectLibrarySnapshot } from '@workbench/projects/library';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  RouterProvider,
} from '@tanstack/react-router';
import { system } from '@theme/system';
import { useWorkbenchPreferences } from '@workbench/settings/store';
import i18n from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const library = vi.hoisted<ProjectLibrarySnapshot>(() => ({
  status: 'ready',
  summaries: [
    {
      createdAt: '2026-09-15T10:00:00.000Z',
      id: 'p-city',
      minimumCanvasSchemaVersion: 3,
      name: 'Neon City Keyframes',
      revision: 1,
      updatedAt: '2026-10-29T12:00:00.000Z',
    },
  ],
}));

vi.mock('@workbench/projects/library', async (importOriginal) => ({
  ...(await importOriginal<typeof libraryModule>()),
  getProjectLibrary: () => library,
  useProjectLibrary: () => library,
}));

import LaunchpadCommandPaletteDialog from './LaunchpadCommandPaletteDialog';
import { SETTINGS_ENTRY_DEPS } from './settingsEntryDeps';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const i18nReady = fetch('/locales/en.json')
  .then((response) => response.json())
  .then((translation) =>
    i18n.use(initReactI18next).init({
      fallbackLng: 'en',
      interpolation: { escapeValue: false },
      lng: 'en',
      resources: { en: { translation } },
    })
  );

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const PaletteHost = () => (
  <>
    <LaunchpadCommandPaletteDialog
      isOpen
      modifierKeyLabel="ctrl"
      preferences={useWorkbenchPreferences()}
      settingsEntryDeps={SETTINGS_ENTRY_DEPS}
      onClose={vi.fn()}
      onExitComplete={vi.fn()}
    />
    <Outlet />
  </>
);

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('LaunchpadCommandPaletteDialog', () => {
  it('finds a project by name and opens it in the editor', async () => {
    await i18nReady;
    const rootRoute = createRootRoute({ component: PaletteHost });
    const router = createRouter({
      history: createMemoryHistory({ initialEntries: ['/projects'] }),
      routeTree: rootRoute.addChildren(
        ['/', '/app', '/projects'].map((path) => createRoute({ getParentRoute: () => rootRoute, path }))
      ),
    });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(async () => {
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <QueryClientProvider client={new QueryClient()}>
              <RouterProvider router={router} />
            </QueryClientProvider>
          </ChakraProvider>
        </I18nextProvider>
      );
      await router.load();
    });

    const input = await vi.waitFor(() => {
      const node = document.querySelector<HTMLInputElement>('[name="command-palette-query"]');
      expect(node).not.toBeNull();
      return node!;
    });
    await act(() => userEvent.fill(input, 'neon city'));
    const option = await vi.waitFor(() => {
      const node = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((candidate) =>
        candidate.textContent?.includes('Neon City Keyframes')
      );
      expect(node).toBeDefined();
      return node!;
    });

    const sectionLabels = [...document.querySelectorAll('[role="presentation"]')].map((node) => node.textContent);
    expect(sectionLabels).toContain('Projects');
    expect(option.textContent).toContain('Edited');
    await act(() => userEvent.click(option));

    await vi.waitFor(() => {
      expect(router.state.location.pathname).toBe('/app');
      expect(router.state.location.search).toEqual({ project: 'p-city' });
    });
  });
});
