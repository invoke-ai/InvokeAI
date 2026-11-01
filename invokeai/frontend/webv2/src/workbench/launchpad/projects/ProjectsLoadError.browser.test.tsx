/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop */
import type { ProjectSummaryDTO } from '@workbench/projects/api';
import type * as projectsApi from '@workbench/projects/api';
import type * as coversModule from '@workbench/projects/covers';

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
import { DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION } from '@workbench/canvasSchemaVersion';
import { HomePage } from '@workbench/launchpad/pages/HomePage';
import { createInstance } from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ProjectsBrowser } from './ProjectsBrowser';

const api = vi.hoisted(() => ({ listProjects: vi.fn() }));

vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  listOpenProjectIds: () => Promise.resolve([]),
  listProjects: api.listProjects,
  setClientStateValue: async () => {},
}));
vi.mock('@workbench/projects/covers', async (importOriginal) => ({
  ...(await importOriginal<typeof coversModule>()),
  loadProjectCovers: () => Promise.resolve(),
}));
vi.mock('@workbench/projects/openProjects', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  refreshOpenProjects: () => Promise.resolve(),
}));
vi.mock('@workbench/projects/useProjectFileActions', () => ({ useImportProjectFile: () => () => undefined }));
vi.mock('@workbench/launchpad/home/LivePanel', () => ({ LivePanel: () => null }));
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuthSession: () => ({ user: null }),
}));
vi.mock('./ProjectActionsMenuHost', () => ({
  useProjectActionsMenu: () => ({ activeProjectId: null, openAtPointer: () => undefined }),
  useProjectActionsMenuTrigger: () => ({ isExpanded: false, onClick: () => undefined, onPointerDown: () => undefined }),
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const PROJECT = {
  board_id: 'board-1',
  created_at: '2026-10-01T00:00:00Z',
  minimum_canvas_schema_version: DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
  minimum_document_schema_version: 4,
  name: 'Portraits',
  project_id: 'project-1',
  revision: 1,
  updated_at: '2026-10-01T00:00:00Z',
  workspace_updated_at: null,
} satisfies ProjectSummaryDTO;

let host: HTMLDivElement;
let root: Root;

const render = async (content: ReactNode) => {
  const rootRoute = createRootRoute({ component: () => <Outlet /> });
  const router = createRouter({
    history: createMemoryHistory({ initialEntries: ['/'] }),
    routeTree: rootRoute.addChildren([
      createRoute({ component: () => content, getParentRoute: () => rootRoute, path: '/' }),
      createRoute({ getParentRoute: () => rootRoute, path: 'app' }),
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
};

const alert = () => page.getByRole('alert');
const emptyState = () => page.getByText(i18n.t('projects.noSavedProjects'), { exact: true });

beforeEach(() => {
  accountLifecycle.activate(`projects-load-error-${Math.random()}`);
  api.listProjects.mockReset().mockRejectedValueOnce(new Error('Database is locked'));
  host = document.createElement('div');
  host.style.height = '800px';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

describe('a failed project listing', () => {
  it('shows only the error on Projects, and Retry brings the projects back with focus on the first', async () => {
    const { refreshProjectLibrary } = await import('@workbench/projects/library');
    await refreshProjectLibrary();
    await render(
      <ProjectsBrowser
        now={Date.parse('2026-10-29T00:00:00Z')}
        pinnedIds={[]}
        searchTerm=""
        sort="edited"
        view="grid"
        onClearSearch={() => undefined}
        onTogglePin={() => undefined}
      />
    );

    await expect
      .poll(() => document.querySelector('[role="alert"]')?.textContent ?? '')
      .toContain('Database is locked');
    expect(emptyState().query()).toBeNull();

    api.listProjects.mockResolvedValueOnce([PROJECT]);
    await userEvent.click(alert().getByRole('button', { name: i18n.t('common.retry') }));

    await expect.element(alert()).not.toBeInTheDocument();
    const open = page.getByRole('link', { name: i18n.t('projects.openProjectLabel', { name: 'Portraits' }) });
    await expect.element(open).toHaveFocus();
  });

  it('shows the error instead of an empty Home, and Retry lands on the project to resume', async () => {
    api.listProjects.mockReset().mockRejectedValue(new Error('Database is locked'));
    await render(<HomePage />);

    await expect
      .poll(() => document.querySelector('[role="alert"]')?.textContent ?? '')
      .toContain('Database is locked');
    expect(page.getByText(i18n.t('launchpad.home.resume')).query()).toBeNull();

    api.listProjects.mockReset().mockResolvedValue([PROJECT]);
    await userEvent.click(alert().getByRole('button', { name: i18n.t('common.retry') }));

    await expect.element(alert()).not.toBeInTheDocument();
    await expect
      .element(page.getByRole('link', { name: i18n.t('projects.openProjectLabel', { name: 'Portraits' }) }).first())
      .toHaveFocus();
  });
});
