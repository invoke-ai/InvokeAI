import { ChakraProvider } from '@chakra-ui/react';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { WorkflowUiAdapter } from './WorkflowUiContext';

import { WorkflowUiProvider } from './WorkflowUiContext';
import { WorkflowDialogHost } from './WorkflowWidgetChrome';

// The dialogs pull in heavy leaf UI (node search, the library browser, graph
// previews) that is irrelevant to the autosave wiring under test here; stub
// them out so this stays a focused wiring-layer test.
vi.mock('./editor/AddNodeDialog', () => ({ AddNodeDialog: () => null }));
vi.mock('./library/WorkflowLibraryDialog', () => ({ WorkflowLibraryDialog: () => null }));
vi.mock('./PendingLibraryWorkflowLoader', () => ({ PendingLibraryWorkflowLoader: () => null }));

const { updateLibraryWorkflowMock } = vi.hoisted(() => ({ updateLibraryWorkflowMock: vi.fn() }));

vi.mock('@features/workflow/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  updateLibraryWorkflow: updateLibraryWorkflowMock,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const createMutablePort = <Snapshot,>(initialSnapshot: Snapshot) => {
  let snapshot = initialSnapshot;
  const listeners = new Set<() => void>();
  return {
    port: {
      getSnapshot: () => snapshot,
      subscribe: (listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    },
    setSnapshot: (next: Snapshot) => {
      snapshot = next;
      for (const listener of listeners) {
        listener();
      }
    },
  };
};

/**
 * Regression coverage for the StrictMode autosaver-disposal bug: the
 * autosaver used to be created once via a `useState` initializer and
 * disposed in a separate effect's cleanup. React StrictMode's dev-only
 * mount→cleanup→mount simulation ran that cleanup without ever re-running
 * the initializer (state is preserved across the simulation, effects are
 * not), permanently disposing the one live instance — autosave then
 * silently no-oped for the rest of the session. The fix creates AND
 * disposes the autosaver within a single mount effect (held in a ref), so
 * the simulation produces a fresh, live instance instead. Mounting under a
 * real `<StrictMode>` here reproduces that simulation; a test that mounted
 * without it would not have caught the bug.
 */
describe('WorkflowDialogHost library autosave under StrictMode', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    updateLibraryWorkflowMock.mockReset();
    updateLibraryWorkflowMock.mockResolvedValue(undefined);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('still autosaves a bound workflow after a graph edit', async () => {
    const boundGraph = { ...createProjectGraph('workflow-1'), libraryWorkflowId: 'library-workflow-1' };
    const project = createMutablePort({
      galleryValues: {},
      graphHistory: [],
      id: 'project-1',
      isWorkflowRunning: false,
      projectGraph: boundGraph,
      workflowValues: {},
    });

    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally stable for this render lifetime
    const adapter = {
      commands: {
        bindLibraryWorkflow: vi.fn(),
        editGraph: vi.fn(),
        redo: vi.fn(),
        replace: vi.fn(),
        restoreSnapshot: vi.fn(),
        saveSnapshot: vi.fn(),
        undo: vi.fn(),
      },
      getProjectGraph: () => project.port.getSnapshot().projectGraph,
      notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
      project: project.port,
      widgets: { open: vi.fn(), patchValues: vi.fn() },
    } as unknown as WorkflowUiAdapter;

    root = createRoot(host);

    await act(() => {
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={adapter}>
              <WorkflowDialogHost />
            </WorkflowUiProvider>
          </ChakraProvider>
        </StrictMode>
      );
    });

    expect(updateLibraryWorkflowMock).not.toHaveBeenCalled();

    // A graph edit on the already-bound project: a new object identity so the
    // dialog host's graph-changed effect fires and schedules the debounced
    // autosave.
    await act(() => {
      project.setSnapshot({
        ...project.port.getSnapshot(),
        projectGraph: { ...boundGraph, name: 'Edited name' },
      });
    });

    // Past the 2s debounce.
    await act(
      () =>
        new Promise<void>((resolve) => {
          setTimeout(resolve, 2100);
        })
    );
    // Let the autosaver's save promise settle.
    await act(
      () =>
        new Promise<void>((resolve) => {
          setTimeout(resolve, 0);
        })
    );

    expect(updateLibraryWorkflowMock).toHaveBeenCalledTimes(1);
    expect(updateLibraryWorkflowMock).toHaveBeenCalledWith(
      'library-workflow-1',
      expect.objectContaining({ name: 'Edited name' })
    );
  });
});
