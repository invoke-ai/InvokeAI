/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { ModelConfig } from '@features/models/core/types';
import type * as ModelsApi from '@features/models/data/api';
import type * as openProjects from '@workbench/projects/openProjects';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { closeModelDetail } from '@features/models/ui/uiStore';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  RouterProvider,
} from '@tanstack/react-router';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

/** The real shell and the real model manager; the other pages and the top bar are stand-ins. */
const pending = vi.hoisted(() => () => new Promise<never>(() => {}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      values ? `${key}:${Object.values(values).map(String).join(',')}` : key,
  }),
}));
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useCapabilities: () => ({ canManageModels: true, canManageNodes: true, canManageUsers: true }),
  UsersPage: () => null,
}));
vi.mock('@features/intermediates', () => ({ INTERMEDIATES_SETTING_ID: 'x', requestIntermediatesFocus: () => {} }));
vi.mock('@features/fonts/launchpad', () => ({ FontsPage: () => null }));
vi.mock('@features/nodes', () => ({ NodesPage: () => null }));
vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<typeof ModelsApi>()),
  getExternalProviderConfigs: pending,
  getFp8StorageSupport: pending,
  getHFTokenStatus: pending,
  getModelsDir: pending,
  getStarterModels: pending,
  listMissingModels: () => Promise.resolve([]),
  listModelInstalls: pending,
  listModels: pending,
}));
vi.mock('@features/models/data/relationshipsApi', () => ({
  addModelRelationship: pending,
  getRelatedModelKeys: () => Promise.resolve([]),
  removeModelRelationship: pending,
}));
vi.mock('@workbench/settings/launchpad', () => ({ PreferencesPage: () => null }));
vi.mock('@workbench/palette/LaunchpadCommandPalette', () => ({ LaunchpadCommandPalette: () => null }));
vi.mock('./LaunchpadTopBar', () => ({ LaunchpadTopBar: () => <div style={{ flexShrink: 0, height: 48 }} /> }));
vi.mock('./pages/HomePage', () => ({ HomePage: () => null }));
vi.mock('./pages/ProjectsPage', () => ({ ProjectsPage: () => null }));
vi.mock('./projects/ProjectActionsMenuHost', () => ({
  ProjectActionsMenuProvider: ({ children }: { children: ReactNode }) => children,
}));
vi.mock('@workbench/projects/openProjects', () => ({
  refreshOpenProjects: () => Promise.resolve(),
  useOpenProjectsSelector: <T,>(select: (value: openProjects.OpenProjectsSnapshot) => T) =>
    select({ activeId: 'one', ids: ['one', 'two'], status: 'ready' } as openProjects.OpenProjectsSnapshot),
}));
vi.mock('@workbench/projects/library', () => ({
  useProjectLibrarySelector: <T,>(select: (value: { summaries: { id: string; name: string }[] }) => T) =>
    select({
      summaries: [
        { id: 'one', name: 'First' },
        { id: 'two', name: 'Second' },
      ],
    }),
}));

const { Launchpad } = await import('./Launchpad');

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const lora = (key: string): ModelConfig =>
  ({
    base: 'sdxl',
    default_settings: null,
    description: null,
    file_size: 1024,
    format: 'lycoris',
    hash: `hash-${key}`,
    key,
    name: `Lora ${key}`,
    path: `/models/${key}.safetensors`,
    source: `/models/${key}.safetensors`,
    source_type: 'path',
    trigger_phrases: [],
    type: 'lora',
  }) as ModelConfig;

let host: HTMLDivElement;
let root: Root;
let queryClient: QueryClient;

const render = async () => {
  const rootRoute = createRootRoute({ component: Launchpad });
  const router = createRouter({
    history: createMemoryHistory({ initialEntries: ['/models'] }),
    routeTree: rootRoute.addChildren(
      ['/', '/app', '/projects', '/models', '/nodes', '/fonts', '/users', '/preferences'].map((path) =>
        createRoute({ getParentRoute: () => rootRoute, path })
      )
    ),
  });

  await act(async () => {
    root.render(
      <QueryClientProvider client={queryClient}>
        <ChakraProvider value={system}>
          <RouterProvider router={router} />
        </ChakraProvider>
      </QueryClientProvider>
    );
    await router.load();
  });
};

const rail = () => page.getByRole('navigation', { name: 'launchpad.sectionsLabel' });
const rect = (element: Element) => element.getBoundingClientRect();
const row = (key: string) => document.querySelector<HTMLElement>(`[data-list-row="${key}"] [data-list-primary]`)!;

/** Inside the window and inside the scroll viewport that holds it, on both axes. */
const isInView = (element: Element) => {
  const bounds = rect(element);
  const viewport = element.closest('[data-scope="scroll-area"][data-part="viewport"]');
  const visible = viewport ? rect(viewport) : new DOMRect(0, 0, innerWidth, innerHeight);

  return (
    bounds.left >= Math.max(0, visible.left) &&
    bounds.top >= Math.max(0, visible.top) &&
    bounds.right <= Math.min(innerWidth, visible.right) &&
    bounds.bottom <= Math.min(innerHeight, visible.bottom)
  );
};

beforeEach(() => {
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  queryClient.setQueryData(['frontend-config'], { show_donation_link: false });
  applyThemeToRoot(DEFAULT_THEME_ID);
  accountLifecycle.activate('launchpad-shell-test', ':user:launchpad-shell-test');
  setModelsSnapshotForTests({ models: [lora('a'), lora('b')], status: 'loaded' });
  closeModelDetail();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  queryClient.clear();
  host.remove();
  accountLifecycle.invalidate();
  await page.viewport(1280, 900);
});

describe('Launchpad shell', () => {
  it('keeps the model manager usable at 720×450 (a 1440×900 window at 200%) beside an icon rail', async () => {
    await page.viewport(720, 450);
    await render();

    const main = document.querySelector('main')!;
    await expect.element(rail()).toBeVisible();
    expect(rect(rail().element()).width).toBeLessThan(64);
    expect(rect(main).height).toBe(450 - 48);
    expect(rect(main).left).toBeGreaterThanOrEqual(0);
    expect(rect(main).right).toBeLessThanOrEqual(innerWidth);
    expect(rect(main).bottom).toBeLessThanOrEqual(innerHeight);
    await expect
      .element(page.getByRole('link', { name: 'launchpad.sections.models' }))
      .toHaveAttribute('aria-current', 'page');

    // The manager loads lazily, as in the app.
    await vi.waitFor(() => expect(row('b')).not.toBeNull(), { timeout: 10_000 });
    await userEvent.click(row('b'));

    const back = page.getByRole('button', { name: 'models.backToList' });
    await expect.element(back).toHaveFocus();
    for (const name of ['common.edit', 'models.actions']) {
      const control = page.getByRole('button', { name, exact: true });
      await expect.element(control).toBeVisible();
      expect(isInView(control.element())).toBe(true);
    }
    expect(isInView(page.getByRole('button', { name: /models.installQueue/ }).element())).toBe(true);
    expect(await auditAccessibility(host)).toEqual([]);

    await back.click();
    await expect.element(page.getByRole('heading', { name: 'models.title' })).toBeVisible();
    expect(document.activeElement).toBe(row('b'));
  });

  it('names every icon in the rail and marks the current project', async () => {
    await page.viewport(720, 450);
    await render();

    const current = page.getByRole('link', { name: 'First · launchpad.openProjects.current' });
    await expect.element(current).toHaveAttribute('data-current-project', '');
    await expect
      .element(page.getByRole('link', { name: 'Second', exact: true }))
      .not.toHaveAttribute('data-current-project');

    await page.getByRole('link', { name: 'launchpad.sections.fonts' }).hover();
    await expect.element(page.getByRole('tooltip')).toHaveTextContent('launchpad.sections.fonts');

    await page.getByRole('button', { name: 'launchpad.help.label' }).click();
    await expect.element(page.getByRole('menuitem', { name: 'launchpad.help.documentation' })).toBeVisible();
    expect(await auditAccessibility(document.querySelector('[role="menu"]')!)).toEqual([]);
    await userEvent.keyboard('{Escape}');
    await expect.element(page.getByRole('button', { name: 'launchpad.help.label' })).toHaveFocus();
  });

  it('shows the labelled rail and both panes when the window widens', async () => {
    await page.viewport(720, 450);
    await render();
    await expect.element(page.getByRole('link', { name: 'launchpad.sections.models' })).toBeVisible();

    await page.viewport(1280, 800);

    await expect.poll(() => rect(rail().element()).width).toBe(224);
    expect(page.getByRole('link', { name: 'launchpad.sections.models' }).element().textContent).toBe(
      'launchpad.sections.models'
    );
    await expect.element(page.getByRole('heading', { name: 'models.title' })).toBeVisible();
    await expect.element(page.getByRole('tablist')).toBeVisible();
  });
});
