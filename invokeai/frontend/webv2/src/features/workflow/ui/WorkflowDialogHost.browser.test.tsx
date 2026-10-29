import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { requestWorkflowRename, workflowUiStore } from '@features/workflow/ui/workflowUiStore';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkflowDialogHost } from './WorkflowWidgetChrome';

// The host's other dialogs and runtimes have their own suites; here only the rename dialog is under test.
vi.mock('./editor/AddNodeDialog', () => ({ AddNodeDialog: () => null }));
vi.mock('./editor/CallSavedWorkflowSyncRuntime', () => ({ CallSavedWorkflowSyncRuntime: () => null }));
vi.mock('./library/WorkflowLibraryDialog', () => ({ WorkflowLibraryDialog: () => null }));
vi.mock('./library/WorkflowPublicationHost', () => ({ WorkflowPublicationHost: () => null }));
vi.mock('./PendingLibraryWorkflowLoader', () => ({ PendingWorkflowLoader: () => null }));

const TRANSLATIONS: Record<string, string> = {
  'workflowLibrary.rename': 'Rename',
  'workflowLibrary.renameTitle': 'Rename workflow',
  'workflowLibrary.workflowName': 'Workflow name',
};

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => TRANSLATIONS[key] ?? key }),
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

const ALPHA: ProjectWorkflowEntry = { document: { ...createProjectGraph('wf-1'), name: 'Alpha' } };
const BETA: ProjectWorkflowEntry = { document: { ...createProjectGraph('wf-2'), name: 'Beta' } };

const projectSnapshot = (workflows: readonly ProjectWorkflowEntry[], activeWorkflowId = workflows[0]!.document.id) => {
  const active = workflows.find((entry) => entry.document.id === activeWorkflowId)!;

  return {
    activeWorkflow: active,
    activeWorkflowId,
    galleryValues: {},
    id: 'project-1',
    isWorkflowRunning: false,
    projectGraph: active.document,
    workflowValues: {},
    workflows,
  };
};

describe('WorkflowDialogHost rename', () => {
  let host: HTMLDivElement;
  let root: Root;
  let project: ReturnType<typeof createMutablePort<ReturnType<typeof projectSnapshot>>>;
  let commands: {
    addWorkflow: ReturnType<typeof vi.fn>;
    editGraph: ReturnType<typeof vi.fn>;
    renameWorkflow: ReturnType<typeof vi.fn>;
  };

  const settleFrame = () =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });
  const flush = () =>
    act(async () => {
      await settleFrame();
      await settleFrame();
    });

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    commands = { addWorkflow: vi.fn(), editGraph: vi.fn(), renameWorkflow: vi.fn() };
    workflowUiStore.patchSnapshot({ renameRequest: null });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    workflowUiStore.patchSnapshot({ renameRequest: null });
  });

  const renderHost = async (workflows: readonly ProjectWorkflowEntry[]) => {
    project = createMutablePort(projectSnapshot(workflows));
    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally stable for this render lifetime
    const adapter = {
      commands,
      getProjectGraph: () => project.port.getSnapshot().projectGraph,
      notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
      project: project.port,
    } as unknown as WorkflowUiAdapter;

    await act(async () => {
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={adapter}>
              <WorkflowDialogHost />
            </WorkflowUiProvider>
          </ChakraProvider>
        </StrictMode>
      );
      await settleFrame();
    });
  };

  const dialog = () =>
    [...document.querySelectorAll<HTMLElement>('[role="dialog"][data-state="open"]')].find(
      (candidate) => candidate.querySelector('h2')?.textContent === 'Rename workflow'
    ) ?? null;
  const input = () => dialog()?.querySelector<HTMLInputElement>('input[name="renameValue"]') ?? null;

  const request = async (workflowId: string) => {
    await act(async () => {
      requestWorkflowRename(workflowId);
      await settleFrame();
    });
    await flush();
  };

  const typeAndSubmit = async (name: string) => {
    const field = input();
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;

    await act(async () => {
      setValue?.call(field, name);
      field?.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });
    await act(async () => {
      field?.closest('form')?.querySelector<HTMLButtonElement>('button[type="submit"]')?.click();
      await settleFrame();
    });
    await flush();
  };

  it('renames the workflow the request named, then closes, and a later request opens again', async () => {
    await renderHost([ALPHA, BETA]);
    expect(dialog()).toBeNull();

    await request('wf-1');
    expect(input()?.value).toBe('Alpha');

    await typeAndSubmit('Alpha, renamed');
    expect(commands.renameWorkflow).toHaveBeenCalledWith('wf-1', 'Alpha, renamed');
    expect(dialog()).toBeNull();

    await request('wf-1');
    expect(dialog()).not.toBeNull();
  });

  it('shows the request only while its workflow is the active one', async () => {
    await renderHost([ALPHA, BETA]);

    await request('wf-1');
    expect(dialog()).not.toBeNull();

    // Another surface switched workflows behind the dialog: the request is for Alpha, so it is no longer shown.
    await act(() => project.setSnapshot(projectSnapshot([ALPHA, BETA], 'wf-2')));
    await flush();
    expect(dialog()).toBeNull();

    // That request is over: Alpha becoming active again does not bring it back.
    await act(() => project.setSnapshot(projectSnapshot([ALPHA, BETA], 'wf-1')));
    await flush();
    expect(dialog()).toBeNull();
    await act(() => project.setSnapshot(projectSnapshot([ALPHA, BETA], 'wf-2')));
    await flush();

    await request('wf-2');
    expect(input()?.value).toBe('Beta');
    await typeAndSubmit('Beta, renamed');
    expect(commands.renameWorkflow).toHaveBeenCalledWith('wf-2', 'Beta, renamed');
    expect(commands.renameWorkflow).not.toHaveBeenCalledWith('wf-1', expect.anything());
  });
});
