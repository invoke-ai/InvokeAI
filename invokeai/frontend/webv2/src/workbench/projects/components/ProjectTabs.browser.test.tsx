/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { Project } from '@workbench/projectContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { getProjectTabId, PROJECT_CONTENT_PANEL_ID } from '@workbench/projects/projectTabsA11y';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => {
  const project = {
    id: 'project-one',
    name: 'Project one',
    queue: { items: [] },
  } as unknown as Project;

  return {
    closeProject: vi.fn(),
    project,
    snapshot: {
      backendConnection: { status: 'connected' },
      projects: [project],
    },
    switchTo: vi.fn(),
  };
});

vi.mock('@features/generation/react', () => ({ flushGenerateDrafts: vi.fn() }));
vi.mock('@features/models', () => ({ useModelLoads: () => [] }));
vi.mock('@features/queue/react', () => ({ useQueueItemProgress: () => null }));
vi.mock('@workbench/projects/projectFile', () => ({ exportOpenProject: vi.fn() }));
vi.mock('@workbench/projects/useProjectActions', () => ({
  useProjectActions: () => ({ closeProject: mocks.closeProject, deleteProject: vi.fn() }),
}));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(mocks.project),
  useWorkbenchCommands: () => ({
    projects: { create: vi.fn(), rename: vi.fn(), switchTo: mocks.switchTo },
  }),
  useWorkbenchQueries: () => ({ getProject: () => mocks.project }),
  useWorkbenchSelector: (selector: (snapshot: typeof mocks.snapshot) => unknown) => selector(mocks.snapshot),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { name?: string }) =>
      key === 'projects.openProjects'
        ? 'Open projects'
        : key === 'projects.closeProjectLabel'
          ? `Close ${options?.name ?? ''}`
          : key,
  }),
}));
vi.mock('./OpenProjectDialog', () => ({ OpenProjectDialog: () => null }));

import { ProjectTabs } from './ProjectTabs';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  mocks.closeProject.mockClear();
  mocks.switchTo.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ProjectTabs accessibility', () => {
  it('keeps actions outside the tablist and closes the focused tab from the keyboard', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <>
            <ProjectTabs />
            <main>
              <div aria-labelledby={getProjectTabId(mocks.project.id)} id={PROJECT_CONTENT_PANEL_ID} role="tabpanel" />
            </main>
          </>
        </ChakraProvider>
      );
    });

    const tabList = host.querySelector<HTMLElement>('[role="tablist"]')!;
    const tab = tabList.querySelector<HTMLButtonElement>('[role="tab"]')!;
    expect(Array.from(tabList.children).every((child) => child.getAttribute('role') === 'tab')).toBe(true);
    expect(tab.getAttribute('aria-controls')).toBe(PROJECT_CONTENT_PANEL_ID);
    expect(document.getElementById(tab.getAttribute('aria-controls')!)).toHaveAttribute('role', 'tabpanel');
    expect(document.getElementById(PROJECT_CONTENT_PANEL_ID)).toHaveAttribute('aria-labelledby', tab.id);
    expect(tab).toHaveAttribute('aria-keyshortcuts', 'Delete Backspace');

    await act(() => tab.focus());
    await act(() => tab.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Delete' })));
    expect(mocks.closeProject).toHaveBeenCalledWith(mocks.project);

    mocks.closeProject.mockClear();
    await act(() => tab.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Backspace' })));
    expect(mocks.closeProject).toHaveBeenCalledWith(mocks.project);

    expect(tab.querySelector('[role="button"]')).toBeNull();
    expect(tabList.querySelectorAll('[role="tab"]')).toHaveLength(1);
  });
});
