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

  it('groups this workflow’s actions apart from import and export, with no library or new workflow', async () => {
    await renderMenu(createProjectGraph('workflow-1'));

    expect(
      Array.from(host.querySelectorAll('[role="group"]'), (group) => group.getAttribute('aria-labelledby'))
    ).toHaveLength(2);
    expect(groupItems('This workflow')).toEqual(['Rename…', 'Details…', 'Save to library…']);
    expect(groupItems('Import and export')).toEqual(['Import from JSON…', 'Export']);
    expect(menuItem('Export').element().getAttribute('aria-haspopup')).toBe('menu');
    // The breadcrumb's switcher carries both now.
    expect(host.textContent).not.toContain('Workflow library…');
    expect(host.textContent).not.toContain('New workflow');
    // Every row leads with its icon.
    for (const item of host.querySelectorAll('[role="menuitem"]')) {
      expect(item.querySelector('svg')).not.toBeNull();
    }
  });

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

  it('opens Export from the keyboard and closes it back to its row', async () => {
    await renderMenu(createProjectGraph('workflow-1'));
    const highlighted = () =>
      Array.from(document.querySelectorAll('[role="menu"] [data-highlighted]'), (item) => item.textContent);

    await act(() => (host.querySelector('[role="menu"]') as HTMLElement).focus());
    for (let step = 0; step < 5; step += 1) {
      await act(() => userEvent.keyboard('{ArrowDown}'));
      await act(() => new Promise(requestAnimationFrame));
    }
    expect(highlighted()).toEqual(['Export']);

    await act(() => userEvent.keyboard('{ArrowRight}'));
    await vi.waitFor(() => expect(highlighted()).toEqual(['Export', 'Export JSON']));
    await act(() => userEvent.keyboard('{ArrowLeft}'));
    await vi.waitFor(() => expect(document.querySelectorAll('[role="menu"]')).toHaveLength(1));
    expect(highlighted()).toEqual(['Export']);
  });

  it('explains why the image or the copy is unavailable instead of acting', async () => {
    clipboard.canCopyText.mockReturnValue(false);
    await renderMenu(createProjectGraph('workflow-1'));
    await openExport();

    // No editor is mounted, as when the panel shows only the form.
    await expect.element(menuItem('Download as PNG')).toHaveAttribute('aria-disabled', 'true');
    await expect
      .element(menuItem('Download as PNG'))
      .toHaveAccessibleDescription('Open the workflow editor to download it as an image');
    await expect.element(menuItem('Copy JSON')).toHaveAttribute('aria-disabled', 'true');
    await act(() => (menuItem('Download as PNG').element() as HTMLElement).click());
    await act(() => (menuItem('Copy JSON').element() as HTMLElement).click());

    expect(imageExport.exportWorkflowAsPng).not.toHaveBeenCalled();
    expect(copyWorkflowJsonMock).not.toHaveBeenCalled();
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

  it('asks the dialog host to import a workflow file', async () => {
    await renderMenu(createProjectGraph('workflow-1'));
    const before = workflowUiStore.getSnapshot().importRequestCount;

    await act(() => userEvent.click(menuItem('Import from JSON…')));

    expect(workflowUiStore.getSnapshot().importRequestCount).toBe(before + 1);
  });

  it('asks the publication host to save the active workflow as a new template', async () => {
    await renderMenu(createProjectGraph('workflow-1'));

    await act(() => userEvent.click(menuItem('Save to library…')));

    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({ kind: 'save-as-new', workflowId: 'workflow-1' });
  });

  it('asks the dialog host to rename the active workflow, naming it', async () => {
    await renderMenu(createProjectGraph('workflow-1'));

    await act(() => userEvent.click(menuItem('Rename…')));

    expect(workflowUiStore.getSnapshot().renameRequest).toMatchObject({ workflowId: 'workflow-1' });
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
  const renderLabel = (source?: ProjectWorkflowSource) =>
    renderProject(projectWithWorkflows([workflow('workflow-1', 'Portraits', source)], 'workflow-1'));
  const PROJECT_WORKFLOWS = [
    workflow('workflow-1', 'Text 2 Image'),
    workflow('workflow-2', 'Upscale'),
    workflow('workflow-3', ''),
    workflow('workflow-4', 'Background swap'),
  ];

  const workflowsButton = () => host.querySelector<HTMLElement>('[data-template-link]')!.closest('button')!;
  const linkDescription = () =>
    document.getElementById(workflowsButton().getAttribute('aria-describedby') ?? '')?.textContent ?? null;
  const menu = () => document.querySelector<HTMLElement>('[role="menu"]');
  const choices = () =>
    Array.from(document.querySelectorAll<HTMLElement>('[role="menuitemradio"]')).map((item) => ({
      checked: item.getAttribute('aria-checked'),
      name: item.textContent,
    }));
  const libraryItem = () => page.getByRole('menuitem', { name: 'Workflow library…' });
  const openMenu = async () => {
    await act(() => userEvent.click(workflowsButton()));
    await vi.waitFor(() => expect(menu()).toBeVisible());
  };
  const expectClosed = () => vi.waitFor(() => expect(menu()).toBeNull());

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

  it('names the current workflow on a menu trigger whose list mounts only while open', async () => {
    await renderProject(projectWithWorkflows(PROJECT_WORKFLOWS, 'workflow-2'));

    await expect
      .element(page.getByRole('button', { name: 'Switch workflow. Current workflow: Upscale' }))
      .toHaveAttribute('aria-haspopup', 'menu');
    expect(workflowsButton()).toHaveAccessibleDescription('Not linked to a library template');
    expect(document.querySelectorAll('[role="menuitemradio"]')).toHaveLength(0);
  });

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

  it('only closes when the active workflow is chosen', async () => {
    await renderProject(projectWithWorkflows(PROJECT_WORKFLOWS, 'workflow-2'));
    await openMenu();

    await act(() => userEvent.click(page.getByRole('menuitemradio', { name: 'Upscale' })));

    await expectClosed();
    expect(commands.selectWorkflow).not.toHaveBeenCalled();
  });

  it('opens the library on the project’s workflows with the active one selected', async () => {
    await renderProject(projectWithWorkflows(PROJECT_WORKFLOWS, 'workflow-2'));
    await openMenu();

    await act(() => userEvent.click(libraryItem()));

    await expectClosed();
    expect(workflowUiStore.getSnapshot()).toMatchObject({
      isLibraryOpen: true,
      librarySelection: { projectId: 'project-1', workflowId: 'workflow-2' },
      libraryTab: 'project',
    });
    expect(commands.selectWorkflow).not.toHaveBeenCalled();
  });

  it('opens with every row, so a click on the library straight away reaches it, on a project of several workflows', async () => {
    await renderProject(projectWithWorkflows(PROJECT_WORKFLOWS, 'workflow-2'));
    await act(() => userEvent.click(workflowsButton()));
    const library = () => document.querySelector<HTMLElement>('[data-value="workflow-library"]')!;
    await vi.waitFor(() => expect(library()).not.toBeNull());
    // Every row is there in the commit that opens the menu; nothing arrives later to push the actions down under the
    // pointer, so a click straight away reaches the library.
    expect(choices()).toHaveLength(PROJECT_WORKFLOWS.length);
    await act(() => userEvent.click(library()));
    expect(workflowUiStore.getSnapshot()).toMatchObject({
      isLibraryOpen: true,
      librarySelection: { projectId: 'project-1', workflowId: 'workflow-2' },
      libraryTab: 'project',
    });
    expect(commands.createWorkflow).not.toHaveBeenCalled();
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

  it('scrolls a long list, opened on the active workflow, under a library item that stays in view', async () => {
    const workflows = Array.from({ length: 60 }, (_, index) => workflow(`workflow-${index}`, `Workflow ${index}`));
    await renderProject(projectWithWorkflows(workflows, 'workflow-45'));
    await openMenu();

    const content = menu()!.getBoundingClientRect();
    const footer = Array.from(menu()!.querySelectorAll<HTMLElement>('[role="menuitem"]'), (item) =>
      item.getBoundingClientRect()
    );
    const viewport = menu()!.querySelector<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"]')!;
    const isInViewport = (name: string) => {
      const item = page.getByRole('menuitemradio', { name }).element().getBoundingClientRect();
      const bounds = viewport.getBoundingClientRect();

      return item.top >= bounds.top - 1 && item.bottom <= bounds.bottom + 1;
    };

    expect(content.bottom).toBeLessThanOrEqual(window.innerHeight);
    expect(viewport.scrollHeight).toBeGreaterThan(viewport.clientHeight);
    expect(footer).toHaveLength(2);
    for (const item of footer) {
      expect(item.top).toBeGreaterThanOrEqual(content.top);
      expect(item.bottom).toBeLessThanOrEqual(content.bottom);
    }
    await vi.waitFor(() => expect(isInViewport('Workflow 45')).toBe(true));

    // The keyboard reaches the end of the list by scrolling it.
    await act(() => userEvent.keyboard('{End}{ArrowUp}{ArrowUp}'));
    await vi.waitFor(() => expect(isInViewport('Workflow 59')).toBe(true));
  });
});
