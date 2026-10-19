import type { Project } from '@workbench/projectContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { reportProjectSync } from '@workbench/projects/syncStore';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const harness = vi.hoisted(() => {
  const state = {
    events: [] as string[],
    exportedPrompt: null as string | null,
    prompt: 'prompt before the debounced edit',
  };
  const makeProject = (): Project =>
    ({
      id: 'project-1',
      name: 'Prompt project',
      queue: { items: [] },
      widgetInstances: {
        generate: { values: { positivePrompt: state.prompt } },
      },
    }) as unknown as Project;

  return {
    ...state,
    flushGenerateDrafts: vi.fn(() => {
      state.events.push('flush');
      state.prompt = 'prompt from the immediate debounced edit';
    }),
    getProject: vi.fn(() => {
      state.events.push('read');
      return makeProject();
    }),
    makeProject,
    startExport: vi.fn((project: Project) => {
      state.events.push('export');
      const widgetInstances = project.widgetInstances as unknown as {
        generate: { values: { positivePrompt: string } };
      };

      state.exportedPrompt = widgetInstances.generate.values.positivePrompt;
    }),
    state,
  };
});

vi.mock('@features/generation/react', () => ({ flushGenerateDrafts: harness.flushGenerateDrafts }));
vi.mock('@features/models', () => ({ useModelLoads: () => [] }));
vi.mock('@features/queue/contracts', () => ({
  getProjectQueueIndicatorState: () => ({ progressState: 'idle', runningQueueItemId: null }),
}));
vi.mock('@features/queue/react', () => ({ useQueueItemProgress: () => null }));
vi.mock('@platform/ui/ConfirmDialog', () => ({ ConfirmDialog: () => null }));
vi.mock('@workbench/components/QueueProgressIndicator', () => ({ QueueCircularProgress: () => null }));
vi.mock('@workbench/launchpad/formatRelativeTime', () => ({ formatRelativeTime: () => '' }));
vi.mock('@workbench/projects/components', () => ({ OpenProjectDialog: () => null }));
vi.mock('@workbench/projects/components/DeleteProjectDialog', () => ({ DeleteProjectDialog: () => null }));
vi.mock('@workbench/projects/library', () => ({
  refreshProjectLibrary: vi.fn(() => Promise.resolve()),
  renameLibraryProject: vi.fn(() => Promise.resolve()),
  useProjectLibrarySelector: (selector: (snapshot: { summaries: [] }) => unknown) => selector({ summaries: [] }),
}));
vi.mock('@workbench/projects/useProjectActions', () => ({
  useProjectActions: () => ({ closeProject: vi.fn(), deleteProject: vi.fn(), openProject: vi.fn() }),
}));
vi.mock('@workbench/projects/useProjectFileActions', () => ({
  useExportOpenProject: () => harness.startExport,
}));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(harness.makeProject()),
  useWorkbenchCommands: () => ({ projects: { create: vi.fn(), rename: vi.fn() } }),
  useWorkbenchQueries: () => ({ getProject: harness.getProject }),
  useWorkbenchSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({
      backendConnection: { status: 'connected' },
      projects: [harness.makeProject(), { id: 'project-2', name: 'Other project', queue: { items: [] } }],
    }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

import { ProjectSwitcher } from './ProjectSwitcher';
import { setProjectSwitcherOpen } from './projectSwitcherStore';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(async () => {
  harness.state.events.length = 0;
  harness.state.exportedPrompt = null;
  harness.state.prompt = 'prompt before the debounced edit';
  harness.flushGenerateDrafts.mockClear();
  harness.getProject.mockClear();
  harness.startExport.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  setProjectSwitcherOpen(true);

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <ProjectSwitcher />
      </ChakraProvider>
    )
  );
});

afterEach(async () => {
  await act(() => {
    setProjectSwitcherOpen(false);
    root?.unmount();
  });
  reportProjectSync({ hasPendingChanges: false, localDraftStatus: 'ok', projects: {}, recoverableDrafts: [] });
  host?.remove();
  host = null;
  root = null;
});

describe('ProjectSwitcher', () => {
  it('marks the open project that awaits a sync decision, and says so on the trigger', async () => {
    await act(() =>
      reportProjectSync({
        hasPendingChanges: true,
        localDraftStatus: 'ok',
        projects: {
          'project-1': { isPendingPush: false, revision: 5 },
          'project-2': {
            conflict: { detectedAt: '2026-10-14T09:00:00.000Z', kind: 'deleted' },
            isPendingPush: true,
            revision: 3,
          },
        },
        recoverableDrafts: [],
      })
    );

    await vi.waitFor(() => {
      const rows = [...document.querySelectorAll('[role="menuitemradio"]')].map((row) => [
        row.getAttribute('data-value'),
        row.textContent?.includes('projects.needsAttention'),
      ]);
      expect(rows).toEqual([
        ['project-1', false],
        ['project-2', true],
      ]);
      expect(
        document.querySelector('button[aria-label="topbar.projectSwitcher.triggerNeedsAttention"]')
      ).not.toBeNull();
    });
  });

  it('keeps the rename dialog open after selecting Rename', async () => {
    let renameItem: HTMLElement | undefined;
    await vi.waitFor(() => {
      renameItem = document.querySelector<HTMLElement>('[role="menuitem"][data-value="rename-project"]') ?? undefined;
      expect(renameItem).toBeDefined();
    });
    await act(() => userEvent.click(renameItem!));
    await vi.waitFor(() => {
      expect(document.querySelector('[role="dialog"][data-state="open"]')).not.toBeNull();
      expect(document.querySelector<HTMLInputElement>('input[name="renameValue"]')?.value).toBe('Prompt project');
    });
    await act(() => userEvent.keyboard('{Escape}'));
    await vi.waitFor(() => {
      expect(document.activeElement).toBe(
        document.querySelector('button[aria-label="topbar.projectSwitcher.trigger"]')
      );
    });
    await vi.waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());
    await act(() => userEvent.keyboard('{Enter}'));
    await vi.waitFor(() => expect(document.querySelector('[role="menu"][data-state="open"]')).not.toBeNull());
    await act(() => document.querySelector<HTMLElement>('[role="menuitem"][data-value="rename-project"]')?.focus());
    await act(() => userEvent.keyboard('{Enter}'));
    await vi.waitFor(() => expect(document.querySelector('[role="dialog"][data-state="open"]')).not.toBeNull());
  });

  it('flushes an immediate debounced prompt edit before reading and exporting the project', async () => {
    let exportItem: HTMLElement | undefined;

    await vi.waitFor(() => {
      exportItem = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
        (item) => item.textContent === 'common.export'
      );
      expect(exportItem).toBeDefined();
    });

    await act(() => userEvent.click(exportItem!));

    expect(harness.state.events).toEqual(['flush', 'read', 'export']);
    expect(harness.state.exportedPrompt).toBe('prompt from the immediate debounced edit');
  });
});
