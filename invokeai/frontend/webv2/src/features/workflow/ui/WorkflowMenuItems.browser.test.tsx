import type { ProjectGraphState, ProjectWorkflowSource, WorkflowNode } from '@features/workflow/core/types';
import type { WorkflowRuntimeApi } from '@features/workflow/ui/contracts';
import type {
  WorkflowProjectPersistence,
  WorkflowReadPort,
  WorkflowUiAdapter,
} from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider, Menu } from '@chakra-ui/react';
import { createProjectGraph } from '@features/workflow/utility';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkflowUiProvider } from './WorkflowUiContext';
import { clearWorkflowPublicationIntent, workflowUiStore } from './workflowUiStore';
import { WorkflowHeaderActions, WorkflowMenuItems, WorkflowWidgetLabel } from './WorkflowWidgetChrome';

const { copyWorkflowJsonMock, downloadWorkflowJsonMock, getLibraryWorkflowRecordMock } = vi.hoisted(() => ({
  copyWorkflowJsonMock: vi.fn(),
  downloadWorkflowJsonMock: vi.fn(),
  getLibraryWorkflowRecordMock: vi.fn(),
}));

vi.mock('@features/workflow/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getLibraryWorkflowRecord: getLibraryWorkflowRecordMock,
}));

const identity = vi.hoisted(() => ({
  session: { multiuserEnabled: false, phase: 'ready', user: null } as {
    multiuserEnabled: boolean;
    phase: 'ready';
    user: { user_id: string } | null;
  },
}));

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuthSession: () => identity.session,
}));

vi.mock('./workflowTransfer', () => ({
  copyWorkflowJson: copyWorkflowJsonMock,
  downloadWorkflowJson: downloadWorkflowJsonMock,
}));

const TRANSLATIONS: Record<string, string> = {
  'projects.exportFailed': 'Export failed',
  'widgets.labels.workflow': 'Workflow',
  'widgets.workflow.addNode': 'Add node',
  'widgets.workflow.copyJson': 'Copy workflow JSON',
  'widgets.workflow.copyJsonFailed': 'Failed to copy workflow JSON',
  'widgets.workflow.detailsWithEllipsis': 'Workflow details…',
  'widgets.workflow.renameWithEllipsis': 'Rename workflow…',
  'widgets.workflow.exportJson': 'Export workflow JSON',
  'widgets.workflow.importJsonWithEllipsis': 'Import workflow JSON…',
  'widgets.workflow.library': 'Workflow library',
  'widgets.workflow.libraryWithEllipsis': 'Workflow library…',
  'widgets.workflow.newWorkflow': 'New workflow',
  'widgets.workflow.persistenceConflict': 'Project has a sync conflict — resolve it to keep saving',
  'widgets.workflow.persistenceError': 'Project save failed — retrying',
  'widgets.workflow.persistencePending': 'Project changes waiting to save (recovery draft kept)',
  'widgets.workflow.persistencePendingNoRecovery': 'Project changes waiting to save — no browser recovery available',
  'widgets.workflow.persistenceSaved': 'Project saved to the server',
  'widgets.workflow.persistenceSaving': 'Saving project…',
  'widgets.workflow.saveToLibraryWithEllipsis': 'Save to library…',
  'widgets.workflow.linkedTemplateUnnamed': 'Linked to a library template',
  'widgets.workflow.untitled': 'Untitled Workflow',
  'widgets.workflow.updateTemplateWithEllipsis': 'Update library template…',
  'workflowLibrary.sourceBundled': 'From a bundled template (read-only)',
  'workflowLibrary.sourceNone': 'Not linked to a library template',
  'workflowLibrary.sourceRevision': 'Linked to library template at revision {{revision}}',
  'workflowLibrary.sourceUnknownRevision': 'Linked to a library template (revision unknown)',
  'workflowLibrary.multipleWorkflowReturnNodesForTransfer':
    'The workflow must contain exactly one workflow return node before it can be exported or copied.',
};

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) =>
      key === 'widgets.workflow.linkedTemplate'
        ? `Linked to library template "${String(options?.name)}"`
        : key === 'workflowLibrary.sourceRevision'
          ? `Linked to library template at revision ${String(options?.revision)}`
          : (TRANSLATIONS[key] ?? key),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const createMutablePort = <Snapshot,>(initialSnapshot: Snapshot) => {
  let snapshot = initialSnapshot;
  const listeners = new Set<() => void>();
  const port: WorkflowReadPort<Snapshot> = {
    getSnapshot: () => snapshot,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  return {
    port,
    setSnapshot: (next: Snapshot) => {
      snapshot = next;
      for (const listener of listeners) {
        listener();
      }
    },
  };
};

const duplicateReturnGraph = (): ProjectGraphState => ({
  ...createProjectGraph('workflow-1'),
  nodes: [
    { data: { type: 'workflow_return' }, id: 'return-1', position: { x: 0, y: 0 }, type: 'invocation' },
    { data: { type: 'workflow_return' }, id: 'return-2', position: { x: 100, y: 0 }, type: 'invocation' },
  ] as unknown as WorkflowNode[],
});

const projectSnapshot = (graph: ProjectGraphState, source?: ProjectWorkflowSource) => ({
  activeWorkflow: { document: graph, source },
  activeWorkflowId: graph.id,
  galleryValues: {},
  id: 'project-1',
  isWorkflowRunning: false,
  projectGraph: graph,
  workflowValues: {},
  workflows: [{ document: graph, source }],
});

const SAVED: WorkflowProjectPersistence = { error: null, hasLocalRecovery: true, lastSavedAt: null, status: 'saved' };

const TEST_RUNTIME: WorkflowRuntimeApi = {
  commands: { register: () => () => undefined },
  hotkeys: { register: () => () => undefined },
  instanceId: 'workflow-menu-test',
  region: 'center',
  typeId: 'workflow',
};

const createCommands = () => ({
  addWorkflow: vi.fn(() => 'workflow-2'),
  createWorkflow: vi.fn(() => 'workflow-2'),
  duplicateWorkflow: vi.fn(() => null),
  editGraph: vi.fn(),
  redo: vi.fn(),
  removeWorkflow: vi.fn(),
  renameWorkflow: vi.fn(),
  selectWorkflow: vi.fn(),
  setWorkflowSource: vi.fn(),
  undo: vi.fn(),
});

describe('WorkflowMenuItems', () => {
  let host: HTMLDivElement;
  let root: Root;
  let notifications: WorkflowUiAdapter['notifications'];
  let commands: ReturnType<typeof createCommands>;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    copyWorkflowJsonMock.mockReset();
    downloadWorkflowJsonMock.mockReset();
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    clearWorkflowPublicationIntent();
    notifications = { error: vi.fn(), info: vi.fn(), success: vi.fn() };
    commands = createCommands();
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    clearWorkflowPublicationIntent();
  });

  const renderMenu = async (graph: ProjectGraphState, source?: ProjectWorkflowSource) => {
    const project = createMutablePort(projectSnapshot(graph, source));
    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally minimal test adapter
    const adapter = {
      commands,
      getProjectGraph: () => graph,
      notifications,
      project: project.port,
      widgets: { open: vi.fn(), patchValues: vi.fn() },
    } as unknown as WorkflowUiAdapter;

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <Menu.Root open>
              <Menu.Positioner>
                <Menu.Content>
                  <WorkflowMenuItems region="center" runtime={TEST_RUNTIME} />
                </Menu.Content>
              </Menu.Positioner>
            </Menu.Root>
          </WorkflowUiProvider>
        </ChakraProvider>
      );
    });
  };

  const menuItems = () => Array.from(host.querySelectorAll<HTMLElement>('[role="menuitem"]'));
  const menuItem = (text: string) => menuItems().find((item) => item.textContent === text);

  it('blocks export and copy with action-specific translated feedback', async () => {
    await renderMenu(duplicateReturnGraph());

    await act(() => menuItem('Export workflow JSON')?.click());
    await act(() => menuItem('Copy workflow JSON')?.click());

    expect(downloadWorkflowJsonMock).not.toHaveBeenCalled();
    expect(copyWorkflowJsonMock).not.toHaveBeenCalled();
    expect(notifications.error).toHaveBeenNthCalledWith(
      1,
      'Export failed',
      'The workflow must contain exactly one workflow return node before it can be exported or copied.'
    );
    expect(notifications.error).toHaveBeenNthCalledWith(
      2,
      'Failed to copy workflow JSON',
      'The workflow must contain exactly one workflow return node before it can be exported or copied.'
    );
  });

  it('starts a new workflow beside the others without asking for confirmation', async () => {
    await renderMenu(createProjectGraph('workflow-1'));

    const item = menuItem('New workflow');
    expect(item).not.toBeUndefined();

    await act(() => item?.click());

    expect(commands.createWorkflow).toHaveBeenCalledTimes(1);
    // Nothing is replaced, so nothing stands between the click and the new workflow.
    expect(document.querySelector('[role="alertdialog"]')).toBeNull();
  });

  it('asks the publication host to save the active workflow as a new template', async () => {
    await renderMenu(createProjectGraph('workflow-1'));

    await act(() => menuItem('Save to library…')?.click());

    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({ kind: 'save-as-new', workflowId: 'workflow-1' });
  });

  it('asks the dialog host to rename the active workflow, naming it', async () => {
    await renderMenu(createProjectGraph('workflow-1'));

    await act(() => menuItem('Rename workflow…')?.click());

    expect(workflowUiStore.getSnapshot().renameRequest).toMatchObject({ workflowId: 'workflow-1' });
  });

  it('offers a template update only for a workflow whose source can be written', async () => {
    await renderMenu(createProjectGraph('workflow-1'));
    expect(menuItem('Update library template…')).toBeUndefined();

    await renderMenu(createProjectGraph('workflow-1'), { libraryWorkflowId: 'default_text', revision: 1 });
    expect(menuItem('Update library template…')).toBeUndefined();

    await renderMenu(createProjectGraph('workflow-1'), { libraryWorkflowId: 'lib-1', revision: 4 });

    const item = menuItem('Update library template…');
    expect(item).not.toBeUndefined();

    await act(() => item?.click());

    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({
      kind: 'update-source',
      workflowId: 'workflow-1',
    });

    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    await renderMenu(createProjectGraph('workflow-1'), {
      libraryWorkflowId: 'lib-public-other-account',
      ownerUserId: 'another-user',
      revision: 4,
    });
    expect(menuItem('Update library template…')).toBeUndefined();
    const saveAsNew = menuItem('Save to library…');
    expect(saveAsNew).not.toBeUndefined();

    await act(() => saveAsNew?.click());

    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({
      kind: 'save-as-new',
      workflowId: 'workflow-1',
    });
  });

  it('matches persisted source ownership against the current account and fails closed without an owner', async () => {
    const source = { libraryWorkflowId: 'lib-1', ownerUserId: 'user-a', revision: 4 };
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    await renderMenu(createProjectGraph('workflow-1'), source);
    expect(menuItem('Update library template…')).not.toBeUndefined();

    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-b' } };
    await renderMenu(createProjectGraph('workflow-1'), source);
    expect(menuItem('Update library template…')).toBeUndefined();
    expect(menuItem('Save to library…')).not.toBeUndefined();

    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    await renderMenu(createProjectGraph('workflow-1'), { libraryWorkflowId: 'lib-1', revision: 4 });
    expect(menuItem('Update library template…')).toBeUndefined();
  });
});

describe('WorkflowHeaderActions', () => {
  let host: HTMLDivElement;
  let root: Root;
  let persistence: ReturnType<typeof createMutablePort<WorkflowProjectPersistence>>;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    clearWorkflowPublicationIntent();
    persistence = createMutablePort<WorkflowProjectPersistence>(SAVED);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    clearWorkflowPublicationIntent();
  });

  const renderHeader = async (source?: ProjectWorkflowSource) => {
    const graph = createProjectGraph('workflow-1');
    const project = createMutablePort(projectSnapshot(graph, source));
    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally minimal test adapter
    const adapter = {
      commands: createCommands(),
      getProjectGraph: () => graph,
      notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
      persistence: persistence.port,
      project: project.port,
      widgets: { open: vi.fn(), patchValues: vi.fn() },
    } as unknown as WorkflowUiAdapter;

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <WorkflowHeaderActions region="center" runtime={TEST_RUNTIME} />
          </WorkflowUiProvider>
        </ChakraProvider>
      );
    });
  };

  const status = () => host.querySelector<HTMLElement>('[role="status"][data-persistence-status]');

  it('names the project persistence state, including a pending save with no local safety net', async () => {
    await renderHeader();

    expect(status()?.dataset.persistenceStatus).toBe('saved');
    expect(status()?.getAttribute('aria-label')).toBe('Project saved to the server');

    const expectations: Array<[WorkflowProjectPersistence['status'], boolean, string]> = [
      ['saving', true, 'Saving project…'],
      ['pending', true, 'Project changes waiting to save (recovery draft kept)'],
      ['pending', false, 'Project changes waiting to save — no browser recovery available'],
      ['error', true, 'Project save failed — retrying'],
      ['conflict', true, 'Project has a sync conflict — resolve it to keep saving'],
      // Recovery only matters while a save is still waiting.
      ['saved', false, 'Project saved to the server'],
    ];

    for (const [state, hasLocalRecovery, label] of expectations) {
      await act(() => persistence.setSnapshot({ ...SAVED, hasLocalRecovery, status: state }));

      expect(status()?.dataset.persistenceStatus).toBe(state);
      expect(status()?.getAttribute('aria-label')).toBe(label);
    }
  });

  it('asks the publication host to save the active workflow as a new template', async () => {
    await renderHeader();

    const button = host.querySelector<HTMLButtonElement>('button[aria-label="Save to library…"]');
    expect(button).not.toBeNull();

    await act(() => button?.click());

    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({ kind: 'save-as-new', workflowId: 'workflow-1' });
  });

  it('offers Update library template only for a template the account can write, and asks for that update', async () => {
    const updateButton = () => host.querySelector<HTMLButtonElement>('button[aria-label="Update library template…"]');

    await renderHeader();
    expect(updateButton()).toBeNull();

    await renderHeader({ libraryWorkflowId: 'default_text-to-image', revision: 1 });
    expect(updateButton()).toBeNull();

    await renderHeader({ libraryWorkflowId: 'library-1', revision: 2 });
    await act(() => updateButton()?.click());

    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({
      kind: 'update-source',
      workflowId: 'workflow-1',
    });
  });
});

describe('WorkflowWidgetLabel', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    getLibraryWorkflowRecordMock.mockReset().mockResolvedValue({ name: 'Portrait template' });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  const renderLabel = async (source?: ProjectWorkflowSource) => {
    const graph = { ...createProjectGraph('workflow-1'), name: 'Portraits' };
    const project = createMutablePort(projectSnapshot(graph, source));
    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally minimal test adapter
    const adapter = { project: project.port } as unknown as WorkflowUiAdapter;

    await act(() => {
      root.render(
        <QueryClientProvider client={new QueryClient()}>
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={adapter}>
              <WorkflowWidgetLabel region="center" />
            </WorkflowUiProvider>
          </ChakraProvider>
        </QueryClientProvider>
      );
    });
  };
  const workflowsButton = () => host.querySelector<HTMLElement>('[data-template-link]')!.closest('button')!;
  const linkDescription = () =>
    document.getElementById(workflowsButton().getAttribute('aria-describedby') ?? '')?.textContent ?? null;

  it('describes the link state on the workflow name: unlinked, bundled, or linked', async () => {
    await renderLabel();
    expect(linkDescription()).toBe('Not linked to a library template');

    await renderLabel({ libraryWorkflowId: 'default_text-to-image', revision: 1 });
    expect(linkDescription()).toBe('From a bundled template (read-only)');

    await renderLabel({ libraryWorkflowId: 'library-1', revision: 2 });
    expect(linkDescription()).toBe('Linked to a library template');
  });

  it('describes another user’s template as read-only rather than bundled', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-b' } };
    await renderLabel({ libraryWorkflowId: 'library-1', ownerUserId: 'user-a', revision: 4 });

    expect(linkDescription()).toBe('Linked to library template at revision 4');
    expect(host.querySelector('[data-template-link]')?.getAttribute('data-template-link')).toBe('read-only');
  });

  it('reads the linked template’s name only once the name’s tooltip is asked for', async () => {
    await renderLabel({ libraryWorkflowId: 'library-1', revision: 2 });
    expect(getLibraryWorkflowRecordMock).not.toHaveBeenCalled();

    await act(() => workflowsButton().focus());

    await vi.waitFor(() => expect(linkDescription()).toBe('Linked to library template "Portrait template"'));
    expect(getLibraryWorkflowRecordMock).toHaveBeenCalledWith('library-1', expect.anything());
  });
});
