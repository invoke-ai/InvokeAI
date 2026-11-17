import type { ProjectGraphState, ProjectWorkflowSource, WorkflowNode } from '@features/workflow/core/types';
import type { WorkflowRegion, WorkflowRuntimeApi } from '@features/workflow/ui/contracts';
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
import { page, userEvent } from 'vitest/browser';

import { registerWorkflowFlowInstance, releaseWorkflowFlowInstance } from './editor/flowInstanceStore';
import { registerWorkflowPngExportHost } from './editor/workflowPngExport';
import { WorkflowUiProvider } from './WorkflowUiContext';
import { clearWorkflowPublicationIntent, setWorkflowLibraryOpen, workflowUiStore } from './workflowUiStore';
import { WorkflowHeaderActions, WorkflowMenuItems, WorkflowWidgetLabel } from './WorkflowWidgetChrome';

// The export's run loads on first use; loading it here keeps that first load out of each test's waits.
await import('./editor/workflowPngExportRun');

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

const imageExport = vi.hoisted(() => ({
  exportWorkflowAsPng: vi.fn(),
  preflightWorkflowImageExport: vi.fn(),
}));

vi.mock('./editor/workflowImageExport', () => imageExport);

const clipboard = vi.hoisted(() => ({ canCopyText: vi.fn(() => true) }));

vi.mock('@platform/browser/clipboard', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  canCopyText: clipboard.canCopyText,
}));

const TRANSLATIONS: Record<string, string> = {
  'projects.exportFailed': 'Export failed',
  'widgets.labels.workflow': 'Workflow',
  'widgets.workflow.addNode': 'Add node',
  'common.copyUnavailable': 'Copying needs HTTPS or localhost',
  'widgets.workflow.copyJson': 'Copy JSON',
  'widgets.workflow.copyJsonFailed': 'Failed to copy workflow JSON',
  'widgets.workflow.detailsWithEllipsis': 'Details…',
  'widgets.workflow.downloadPng': 'Download as PNG',
  'widgets.workflow.downloadPngUnavailable': 'Open the workflow editor to download it as an image',
  'widgets.workflow.downloadPngRunning': 'Already downloading the workflow as an image',
  'widgets.workflow.export': 'Export',
  'widgets.workflow.exportJson': 'Export JSON',
  'widgets.workflow.importJsonWithEllipsis': 'Import from JSON…',
  'widgets.workflow.menuImportExport': 'Import and export',
  'widgets.workflow.menuThisWorkflow': 'This workflow',
  'widgets.workflow.renameWithEllipsis': 'Rename…',
  'widgets.workflow.saveAsNewTemplateWithEllipsis': 'Save as new template…',
  'widgets.workflow.inThisProject': 'In this project',
  'widgets.workflow.library': 'Workflow library',
  'widgets.workflow.libraryWithEllipsis': 'Workflow library…',
  'widgets.workflow.newWorkflow': 'New workflow',
  'widgets.workflow.persistenceConflict': 'Project has a sync conflict — resolve it to keep saving',
  'widgets.workflow.persistenceError': 'Project save failed — retrying',
  'widgets.workflow.persistencePending': 'Project changes waiting to save (recovery draft kept)',
  'widgets.workflow.persistencePendingNoRecovery': 'Project changes waiting to save — no browser recovery available',
  'widgets.workflow.persistenceSaved': 'Project saved to the server',
  'widgets.workflow.persistenceSaving': 'Saving project…',
  'widgets.workflow.toolbar.redo': 'Redo',
  'widgets.workflow.toolbar.undo': 'Undo',
  'widgets.workflow.saveToLibraryWithEllipsis': 'Save to library…',
  'widgets.workflow.switchWorkflow': 'Switch workflow',
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
          : key === 'widgets.workflow.switchWorkflowTrigger'
            ? `Switch workflow. Current workflow: ${String(options?.name)}`
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

interface HistoryState {
  canRedo: boolean;
  canUndo: boolean;
}

const NO_HISTORY: HistoryState = { canRedo: false, canUndo: false };

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
  const releases: (() => void)[] = [];

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    copyWorkflowJsonMock.mockReset();
    downloadWorkflowJsonMock.mockReset();
    imageExport.exportWorkflowAsPng.mockReset().mockResolvedValue({ status: 'exported', reduced: false });
    imageExport.preflightWorkflowImageExport.mockReset().mockReturnValue(null);
    clipboard.canCopyText.mockReturnValue(true);
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    clearWorkflowPublicationIntent();
    notifications = { error: vi.fn(), info: vi.fn(), success: vi.fn() };
    commands = createCommands();
  });

  afterEach(async () => {
    for (const release of releases.splice(0)) {
      release();
    }
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
  /** A mounted editor, as far as the PNG export needs one: its offscreen view and its flow instance. */
  const mountEditor = () => {
    const flowElement = document.createElement('div');
    const flow = {
      getNodes: () => [{ id: 'node-1' }],
      getNodesBounds: () => ({ height: 10, width: 10, x: 0, y: 0 }),
    } as unknown as Parameters<typeof registerWorkflowFlowInstance>[0];
    const prepare = vi.fn(() => Promise.resolve(flowElement));
    const complete = vi.fn();

    registerWorkflowFlowInstance(flow);
    releases.push(registerWorkflowPngExportHost({ complete, prepare }), () => releaseWorkflowFlowInstance(flow));

    return { complete, flowElement, prepare };
  };

  const menuItem = (name: string) => page.getByRole('menuitem', { exact: true, name });
  const groupItems = (name: string) =>
    Array.from(
      page.getByRole('group', { name }).element().querySelectorAll('[role="menuitem"]'),
      (item) => item.textContent
    );
  const openExport = async () => {
    await act(() => userEvent.click(menuItem('Export')));
    await expect.element(menuItem('Export JSON')).toBeVisible();
  };

  it('exports, copies and draws the workflow from the Export submenu through the existing commands', async () => {
    const graph = { ...createProjectGraph('workflow-1'), name: 'Portraits' };
    const editor = mountEditor();
    copyWorkflowJsonMock.mockResolvedValue(undefined);
    await renderMenu(graph);

    await openExport();
    expect(
      Array.from(
        document.querySelectorAll('[role="menu"]')[1]!.querySelectorAll('[role="menuitem"]'),
        (item) => item.textContent
      )
    ).toEqual(['Export JSON', 'Copy JSON', 'Download as PNG']);
    await act(() => userEvent.click(menuItem('Export JSON')));
    expect(downloadWorkflowJsonMock).toHaveBeenCalledExactlyOnceWith(graph);

    await openExport();
    await act(() => userEvent.click(menuItem('Copy JSON')));
    expect(copyWorkflowJsonMock).toHaveBeenCalledExactlyOnceWith(graph);
    await vi.waitFor(() => expect(notifications.success).toHaveBeenCalledOnce());

    await openExport();
    await act(() => userEvent.click(menuItem('Download as PNG')));
    await vi.waitFor(() => expect(editor.complete).toHaveBeenCalledOnce());
    expect(imageExport.exportWorkflowAsPng).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ flowElement: editor.flowElement, workflowName: 'Portraits' })
    );
    expect(notifications.error).not.toHaveBeenCalled();
  });

  it('blocks export and copy with action-specific translated feedback', async () => {
    await renderMenu(duplicateReturnGraph());

    await openExport();
    await act(() => userEvent.click(menuItem('Export JSON')));
    await openExport();
    await act(() => userEvent.click(menuItem('Copy JSON')));

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

  it('offers a template update only for a workflow whose source can be written, and then names the save for what it makes', async () => {
    const thisWorkflow = () => groupItems('This workflow');

    await renderMenu(createProjectGraph('workflow-1'));
    expect(thisWorkflow()).toEqual(['Rename…', 'Details…', 'Save to library…']);

    await renderMenu(createProjectGraph('workflow-1'), { libraryWorkflowId: 'default_text', revision: 1 });
    expect(thisWorkflow()).toEqual(['Rename…', 'Details…', 'Save to library…']);

    await renderMenu(createProjectGraph('workflow-1'), { libraryWorkflowId: 'lib-1', revision: 4 });
    expect(thisWorkflow()).toEqual(['Rename…', 'Details…', 'Save as new template…', 'Update library template…']);

    await act(() => userEvent.click(menuItem('Update library template…')));
    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({
      kind: 'update-source',
      workflowId: 'workflow-1',
    });
    // Saving a linked workflow still makes a new template; only its label changes.
    await act(() => userEvent.click(menuItem('Save as new template…')));
    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({ kind: 'save-as-new', workflowId: 'workflow-1' });

    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    await renderMenu(createProjectGraph('workflow-1'), {
      libraryWorkflowId: 'lib-public-other-account',
      ownerUserId: 'another-user',
      revision: 4,
    });
    expect(thisWorkflow()).toEqual(['Rename…', 'Details…', 'Save to library…']);
  });

  it('matches persisted source ownership against the current account and fails closed without an owner', async () => {
    const source = { libraryWorkflowId: 'lib-1', ownerUserId: 'user-a', revision: 4 };
    const canUpdate = () => groupItems('This workflow').includes('Update library template…');
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    await renderMenu(createProjectGraph('workflow-1'), source);
    expect(canUpdate()).toBe(true);

    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-b' } };
    await renderMenu(createProjectGraph('workflow-1'), source);
    expect(canUpdate()).toBe(false);

    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    await renderMenu(createProjectGraph('workflow-1'), { libraryWorkflowId: 'lib-1', revision: 4 });
    expect(canUpdate()).toBe(false);
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

  let commands: ReturnType<typeof createCommands>;
  let project: ReturnType<typeof createMutablePort<ReturnType<typeof projectSnapshot> & HistoryState>>;

  const renderHeader = async (
    source?: ProjectWorkflowSource,
    { history = NO_HISTORY, region = 'center' }: { history?: HistoryState; region?: WorkflowRegion } = {}
  ) => {
    const graph = createProjectGraph('workflow-1');
    project = createMutablePort({ ...projectSnapshot(graph, source), ...history });
    commands = createCommands();
    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally minimal test adapter
    const adapter = {
      CommandShortcut: ({ commandId }: { commandId: string }) => <kbd>{commandId}</kbd>,
      commands,
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
            <WorkflowHeaderActions region={region} runtime={TEST_RUNTIME} />
          </WorkflowUiProvider>
        </ChakraProvider>
      );
    });
  };

  const actionLabels = () =>
    Array.from(host.querySelectorAll<HTMLElement>('button, [role="status"]'), (action) =>
      action.getAttribute('aria-label')
    );
  const historyButton = (label: string) => host.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);

  it('steps the editor history from the header only where the editor is, and only while there is a step', async () => {
    await renderHeader();
    expect(actionLabels()).toEqual(['Add node', 'Undo', 'Redo', 'Save to library…', 'Project saved to the server']);
    expect(historyButton('Undo')?.getAttribute('aria-disabled')).toBe('true');
    expect(historyButton('Redo')?.getAttribute('aria-disabled')).toBe('true');

    await act(() => historyButton('Undo')?.click());
    await act(() => historyButton('Redo')?.click());
    expect(commands.undo).not.toHaveBeenCalled();
    expect(commands.redo).not.toHaveBeenCalled();

    await act(() => project.setSnapshot({ ...project.port.getSnapshot(), canRedo: true, canUndo: true }));
    expect(historyButton('Undo')?.hasAttribute('aria-disabled')).toBe(false);
    await act(() => historyButton('Undo')?.click());
    await act(() => historyButton('Redo')?.click());
    // The click event must not reach the command as a target: no argument means the active workflow.
    expect(commands.undo).toHaveBeenCalledExactlyOnceWith();
    expect(commands.redo).toHaveBeenCalledExactlyOnceWith();

    // The expanded bottom panel shows the editor too; the left region's linear form has no history to step.
    await renderHeader(undefined, { region: 'bottom' });
    expect(actionLabels()).toEqual(['Undo', 'Redo', 'Workflow library']);
    await renderHeader(undefined, { region: 'left' });
    expect(actionLabels()).toEqual(['Workflow library']);
  });

  it('keeps keyboard focus on Undo when it steps back over the last edit', async () => {
    await renderHeader(undefined, { history: { canRedo: false, canUndo: true } });
    const undo = historyButton('Undo')!;
    undo.focus();
    await act(() => undo.click());
    // The history is now empty: the button says so but keeps focus, so the keyboard user is not dropped to the page.
    await act(() => project.setSnapshot({ ...project.port.getSnapshot(), canRedo: true, canUndo: false }));

    expect(commands.undo).toHaveBeenCalledOnce();
    expect(undo.getAttribute('aria-disabled')).toBe('true');
    expect(document.activeElement).toBe(undo);
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
  let commands: ReturnType<typeof createCommands>;
  let project: ReturnType<typeof createMutablePort<ReturnType<typeof projectWithWorkflows>>>;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    getLibraryWorkflowRecordMock.mockReset().mockResolvedValue({ name: 'Portrait template' });
    commands = createCommands();
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    setWorkflowLibraryOpen(false);
  });

  const workflow = (id: string, name: string, source?: ProjectWorkflowSource) => ({
    document: { ...createProjectGraph(id), name },
    source,
  });
  const projectWithWorkflows = (workflows: ReturnType<typeof workflow>[], activeWorkflowId: string) => {
    const activeWorkflow = workflows.find((entry) => entry.document.id === activeWorkflowId)!;

    return {
      activeWorkflow,
      activeWorkflowId,
      galleryValues: {},
      id: 'project-1',
      isWorkflowRunning: false,
      projectGraph: activeWorkflow.document,
      workflowValues: {},
      workflows,
    };
  };

  const renderProject = async (snapshot: ReturnType<typeof projectWithWorkflows>) => {
    project = createMutablePort(snapshot);
    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally minimal test adapter
    const adapter = { commands, project: project.port } as unknown as WorkflowUiAdapter;

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

  const PROJECT_WORKFLOWS = [
    workflow('workflow-1', 'Text 2 Image'),
    workflow('workflow-2', 'Upscale'),
    workflow('workflow-3', ''),
    workflow('workflow-4', 'Background swap'),
  ];

  const workflowsButton = () => host.querySelector<HTMLElement>('[data-template-link]')!.closest('button')!;

  const menu = () => document.querySelector<HTMLElement>('[role="menu"]');
  const choices = () =>
    Array.from(document.querySelectorAll<HTMLElement>('[role="menuitemradio"]')).map((item) => ({
      checked: item.getAttribute('aria-checked'),
      name: item.textContent,
    }));

  const openMenu = async () => {
    await act(() => userEvent.click(workflowsButton()));
    await vi.waitFor(() => expect(menu()).toBeVisible());
  };
  const expectClosed = () => vi.waitFor(() => expect(menu()).toBeNull());

  it('lists the project’s workflows in project order with the active one checked', async () => {
    await renderProject(projectWithWorkflows(PROJECT_WORKFLOWS, 'workflow-2'));
    await openMenu();

    expect(choices()).toEqual([
      { checked: 'false', name: 'Text 2 Image' },
      { checked: 'true', name: 'Upscale' },
      { checked: 'false', name: 'Untitled Workflow' },
      { checked: 'false', name: 'Background swap' },
    ]);
    await expect.element(page.getByRole('group', { name: 'In this project' })).toBeVisible();
    // New workflow and the library follow, past a separator.
    const items = menu()!.querySelectorAll('[role="menuitemradio"], [role="menuitem"], [role="separator"]');
    expect(Array.from(items, (item) => item.textContent || item.getAttribute('role')).slice(-3)).toEqual([
      'separator',
      'New workflow',
      'Workflow library…',
    ]);
  });

  it('opens another workflow through the project’s select command and closes', async () => {
    await renderProject(projectWithWorkflows(PROJECT_WORKFLOWS, 'workflow-2'));
    await openMenu();

    await act(() => userEvent.click(page.getByRole('menuitemradio', { name: 'Background swap' })));

    expect(commands.selectWorkflow).toHaveBeenCalledExactlyOnceWith('workflow-4');
    await expectClosed();
    expect(workflowUiStore.getSnapshot().isLibraryOpen).toBe(false);
  });

  it('adds a new workflow that becomes the active one, returning focus to the renamed trigger', async () => {
    const snapshot = projectWithWorkflows(PROJECT_WORKFLOWS, 'workflow-2');
    commands.createWorkflow.mockImplementation(() => {
      const created = workflow('workflow-new', 'Fresh workflow');
      project.setSnapshot(projectWithWorkflows([...snapshot.workflows, created], created.document.id));
      return created.document.id;
    });
    await renderProject(snapshot);
    await openMenu();

    const newWorkflow = page.getByRole('menuitem', { name: 'New workflow' });
    // A pointer arriving over an item refocuses the menu a frame later; a click in that same frame would lose the
    // restored focus to it, which a person's pointer never does.
    await act(() => userEvent.hover(newWorkflow));
    await act(() => new Promise(requestAnimationFrame));
    await act(() => userEvent.click(newWorkflow));

    expect(commands.createWorkflow).toHaveBeenCalledOnce();
    expect(commands.selectWorkflow).not.toHaveBeenCalled();
    await expectClosed();
    await vi.waitFor(() =>
      expect(document.activeElement?.getAttribute('aria-label')).toBe(
        'Switch workflow. Current workflow: Fresh workflow'
      )
    );
    expect(document.activeElement).toBe(workflowsButton());
    await openMenu();
    expect(choices().at(-1)).toEqual({ checked: 'true', name: 'Fresh workflow' });
  });

  it('opens from the keyboard, moves by arrows and typeahead, and returns focus on Escape', async () => {
    await renderProject(projectWithWorkflows(PROJECT_WORKFLOWS, 'workflow-2'));
    const highlighted = () => document.querySelector('[role="menu"] [data-highlighted]')?.textContent;

    for (const key of ['{Enter}', ' ', '{ArrowDown}']) {
      await act(() => workflowsButton().focus());
      await act(() => userEvent.keyboard(key));
      await vi.waitFor(() => expect(menu()).toBeVisible());
      expect(workflowsButton().getAttribute('aria-expanded')).toBe('true');

      await act(() => userEvent.keyboard('{Escape}'));
      await expectClosed();
      expect(document.activeElement).toBe(workflowsButton());
    }

    await act(() => userEvent.keyboard('{ArrowDown}'));
    // Zag moves focus into the open menu a frame later; keys sent before then reach the trigger.
    await vi.waitFor(() => expect(document.activeElement).toBe(menu()));
    expect(highlighted()).toBe('Text 2 Image');
    await act(() => userEvent.keyboard('b'));
    await vi.waitFor(() => expect(highlighted()).toBe('Background swap'));
    await act(() => userEvent.keyboard('{ArrowDown}'));
    await vi.waitFor(() => expect(highlighted()).toBe('New workflow'));
    await act(() => userEvent.keyboard('{ArrowDown}'));
    await vi.waitFor(() => expect(highlighted()).toBe('Workflow library…'));
    await act(() => userEvent.keyboard('{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}'));
    await vi.waitFor(() => expect(highlighted()).toBe('Upscale'));
    await act(() => userEvent.keyboard('{ArrowUp}'));
    await vi.waitFor(() => expect(highlighted()).toBe('Text 2 Image'));
    // An arrow refocuses the menu a frame later; Enter in that same frame would lose the restored focus to it.
    await act(() => new Promise(requestAnimationFrame));
    await act(() => userEvent.keyboard('{Enter}'));

    expect(commands.selectWorkflow).toHaveBeenCalledExactlyOnceWith('workflow-1');
    await expectClosed();
    expect(document.activeElement).toBe(workflowsButton());
  });

  it('follows workflows added, renamed, reordered and removed while open', async () => {
    const [first, second] = PROJECT_WORKFLOWS;
    await renderProject(projectWithWorkflows([first!, second!], 'workflow-1'));
    await openMenu();
    const names = () => choices().map((choice) => choice.name);

    await act(() =>
      project.setSnapshot(
        projectWithWorkflows(
          [workflow('workflow-2', 'Upscale 4×'), first!, workflow('workflow-5', 'Inpaint')],
          'workflow-1'
        )
      )
    );
    expect(names()).toEqual(['Upscale 4×', 'Text 2 Image', 'Inpaint']);

    await act(() => project.setSnapshot(projectWithWorkflows([workflow('workflow-5', 'Inpaint')], 'workflow-5')));
    expect(choices()).toEqual([{ checked: 'true', name: 'Inpaint' }]);
  });
});
