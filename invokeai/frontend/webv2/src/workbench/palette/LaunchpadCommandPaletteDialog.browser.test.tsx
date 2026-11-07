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

// A fixture index, so retitling the real docs cannot break this test; the provider still loads it through import().
vi.mock('./docsSearchIndex.json', () => ({
  default: {
    pages: [
      { c: 0, h: [['launcher', 'Launcher']], s: 'start-here/quick-setup', t: 'Quick Setup Guide' },
      { c: 1, s: 'troubleshooting/faq', t: 'FAQ' },
    ],
    topics: [
      { d: 'Install, requirements', l: 'Get Started', s: 'start-here/quick-setup' },
      { d: 'FAQ, known bugs', l: 'Troubleshooting', s: 'troubleshooting/faq' },
    ],
  },
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

const PaletteHost = ({ onClose }: { onClose: () => void }) => (
  <>
    <LaunchpadCommandPaletteDialog
      isOpen
      modifierKeyLabel="ctrl"
      preferences={useWorkbenchPreferences()}
      settingsEntryDeps={SETTINGS_ENTRY_DEPS}
      onClose={onClose}
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
  vi.restoreAllMocks();
});

const renderLaunchpadPalette = async (onClose: () => void = vi.fn()) => {
  await i18nReady;
  const rootRoute = createRootRoute({ component: () => <PaletteHost onClose={onClose} /> });
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
  return { input, router };
};

const docsOptions = () =>
  [...document.querySelectorAll<HTMLElement>('[role="option"]')].filter((option) =>
    option.textContent?.includes('Opens in a new tab')
  );

describe('LaunchpadCommandPaletteDialog', () => {
  it('finds a project by name and opens it in the editor', async () => {
    const { input, router } = await renderLaunchpadPalette();
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

  it('lists docs pages under Docs, marks them as opening a new tab, and opens the page on Enter', async () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null);
    const onClose = vi.fn();
    const { input } = await renderLaunchpadPalette(onClose);

    await act(() => userEvent.fill(input, 'quick setup'));
    const option = await vi.waitFor(
      () => {
        const labels = [...document.querySelectorAll('[role="presentation"]')].map((node) => node.textContent);
        expect(labels).toContain('Docs');
        const node = docsOptions().find((candidate) => candidate.textContent?.startsWith('Quick Setup Guide'));
        expect(node).toBeDefined();
        return node!;
      },
      { timeout: 5000 }
    );

    // The cue is text in the option's accessible name, not only the icon.
    expect(option.querySelector('svg')?.getAttribute('aria-hidden')).toBe('true');

    for (let step = 0; step < 20 && input.getAttribute('aria-activedescendant') !== option.id; step += 1) {
      await act(() => userEvent.keyboard('{ArrowDown}'));
    }
    expect(input.getAttribute('aria-activedescendant')).toBe(option.id);
    await act(() => userEvent.keyboard('{Enter}'));

    expect(open).toHaveBeenCalledWith('https://v7.invoke.ai/start-here/quick-setup/', '_blank', 'noopener,noreferrer');
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('adds nothing on open, and browses the topics when the Docs scope opens empty', async () => {
    const { input } = await renderLaunchpadPalette();
    const scopeCommand = await vi.waitFor(() => {
      const node = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find(
        (option) => option.textContent === 'Search docs…'
      );
      expect(node).toBeDefined();
      return node!;
    });
    expect(docsOptions()).toEqual([]);

    await act(() => userEvent.click(scopeCommand));

    await vi.waitFor(
      () =>
        expect(docsOptions().map((option) => option.textContent)).toEqual([
          'Get StartedInstall, requirementsOpens in a new tab',
          'TroubleshootingFAQ, known bugsOpens in a new tab',
        ]),
      { timeout: 5000 }
    );
    expect(document.body.textContent).not.toContain('No Docs yet');
    expect(input.value).toBe('');
  });
});
