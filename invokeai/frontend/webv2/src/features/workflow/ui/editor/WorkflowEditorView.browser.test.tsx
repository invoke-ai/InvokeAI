import type { ProjectGraphState, WorkflowInvocationNode } from '@features/workflow/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

/* eslint-disable react-perf/jsx-no-new-object-as-prop -- each test mounts a fresh editor on purpose */
import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { createProjectGraph } from '@features/workflow/utility';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { commands, userEvent } from 'vitest/browser';
import '@xyflow/react/dist/style.css';

import { clearWorkflowClipboard, copyNodesToClipboard } from './clipboard';
import { getWorkflowFlowInstance } from './flowInstanceStore';
import { CONTENT_VISIBILITY_ZOOM } from './InvocationFlowNode';
import { WorkflowEditorView } from './WorkflowEditorView';
import { clearWorkflowViewports, getWorkflowViewport, getWorkflowViewportKey } from './workflowViewportStore';

const templatesSnapshot = vi.hoisted(() => {
  const field = { batch: false, cardinality: 'SINGLE', name: 'IntegerField' } as const;

  return {
    status: 'loaded',
    templates: {
      add: {
        category: 'math',
        classification: 'stable',
        description: '',
        inputs: {
          a: {
            default: 0,
            description: '',
            exclusiveMaximum: null,
            exclusiveMinimum: null,
            fieldKind: 'input',
            input: 'any',
            maximum: null,
            minimum: null,
            multipleOf: null,
            name: 'a',
            options: null,
            required: false,
            title: 'A',
            type: field,
            uiChoiceLabels: null,
            uiComponent: null,
            uiHidden: false,
            uiModelBase: null,
            uiModelFormat: null,
            uiModelType: null,
            uiOrder: null,
          },
        },
        nodePack: 'invokeai',
        outputType: 'integer_output',
        outputs: { value: { description: '', name: 'value', title: 'Value', type: field } },
        tags: [],
        title: 'Add',
        type: 'add',
        useCache: true,
        version: '1.0.0',
      },
    },
  };
});

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureInvocationTemplatesLoaded: () => {},
  getInvocationTemplatesSnapshot: () => templatesSnapshot,
  subscribeInvocationTemplates: () => () => {},
  useInvocationTemplatesSelector: (selector: (snapshot: unknown) => unknown) => selector(templatesSnapshot),
}));

declare module 'vitest/browser' {
  interface BrowserCommands {
    mouseDragPath: (points: readonly { x: number; y: number }[], frameWidth: number) => Promise<void>;
  }
}

const exportMocks = vi.hoisted(() => ({ rasterize: vi.fn() }));
vi.mock('./workflowImageRaster', () => ({ rasterizeWorkflowImage: exportMocks.rasterize }));
vi.mock('@platform/browser/downloadBlob', () => ({ downloadBlob: vi.fn() }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: {} } },
});

const PROJECT_ID = 'project-1';
const INSTANCE_ID = 'workflow:center';

const addNode = (id: string, x: number, y: number): WorkflowInvocationNode => ({
  data: {
    inputs: { a: { label: '', name: 'a', value: 0 } },
    isIntermediate: true,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type: 'add',
    useCache: true,
    version: '1.0.0',
  },
  id,
  position: { x, y },
  type: 'invocation',
});

const projectSnapshotFor = (graph: ProjectGraphState) => ({
  activeWorkflow: { document: graph },
  activeWorkflowId: graph.id,
  galleryValues: {},
  id: PROJECT_ID,
  isWorkflowRunning: false,
  projectGraph: graph,
  workflowValues: {},
  workflows: [{ document: graph }],
});

/** The editor's project port over a graph the test can replace, as switching the active workflow does. */
const createProjectStore = (graph: ProjectGraphState) => {
  let snapshot = projectSnapshotFor(graph);
  const listeners = new Set<() => void>();

  return {
    getSnapshot: () => snapshot,
    setGraph: (next: ProjectGraphState) => {
      snapshot = projectSnapshotFor(next);
      listeners.forEach((listener) => listener());
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);

      return () => listeners.delete(listener);
    },
  };
};

const createAdapter = (
  project: ReturnType<typeof createProjectStore>,
  {
    nodeExecution,
    showMinimap = false,
  }: { nodeExecution?: WorkflowUiAdapter['nodeExecution']; showMinimap?: boolean } = {}
): WorkflowUiAdapter => {
  const subscribe = () => () => {};

  return {
    CommandShortcut: () => null,
    capabilities: { getSnapshot: () => ({ canUseCache: true }), subscribe },
    commands: {
      addWorkflow: vi.fn(),
      createWorkflow: vi.fn(),
      duplicateWorkflow: vi.fn(),
      editGraph: vi.fn(),
      redo: vi.fn(),
      removeWorkflow: vi.fn(),
      renameWorkflow: vi.fn(),
      selectWorkflow: vi.fn(),
      setWorkflowSource: vi.fn(),
      undo: vi.fn(),
    },
    getProjectGraph: () => project.getSnapshot().projectGraph,
    nodeExecution: nodeExecution ?? { get: () => null, getOrigin: () => null, subscribe, subscribeOrigin: subscribe },
    notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
    openAddModels: vi.fn(),
    performance: {
      mark: vi.fn(),
      measure: vi.fn(),
      time: <T,>(_name: string, _source: unknown, callback: () => T) => callback(),
    },
    preferences: {
      getSnapshot: () => ({
        reduceMotion: true,
        themeId: 'classic',
        workflowEdgeStyle: 'curved',
        workflowEdgesBehindNodes: false,
        workflowShowMinimap: showMinimap,
        workflowSnapToGrid: false,
        workflowValidateConnections: true,
      }),
      subscribe,
    },
    project: { getSnapshot: project.getSnapshot, subscribe: project.subscribe },
    widgets: { open: vi.fn(), patchValues: vi.fn() },
  } as unknown as WorkflowUiAdapter;
};

/** The editor's registered command handlers, so a test can run a command as the palette or a hotkey would. */
const commandHandlers = new Map<string, () => void>();

const runtime = {
  commands: {
    register: ({ handler, id }: { handler: () => void; id: string }) => {
      commandHandlers.set(id, handler);
      return () => commandHandlers.delete(id);
    },
  },
  hotkeys: { register: () => () => {} },
  instanceId: INSTANCE_ID,
  region: 'center',
  typeId: 'workflow',
} as const;

/** Mounts the editor over the project port and returns the adapter, whose commands the test inspects. */
const renderEditor = async (
  root: Root,
  project: ReturnType<typeof createProjectStore>,
  options?: Parameters<typeof createAdapter>[1]
) => {
  const adapter = createAdapter(project, options);

  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={new QueryClient()}>
            <WorkflowUiProvider adapter={adapter}>
              <WorkflowEditorView runtime={runtime} />
            </WorkflowUiProvider>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );

  return adapter;
};

describe('WorkflowEditorView first open', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.cssText = 'width: 480px; height: 520px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    clearWorkflowViewports();
  });

  /** Opens the graph, waits for its mount fit, and counts every field input ever added to a node. */
  const openFitted = async (graph: ProjectGraphState) => {
    let addedFieldInputs = 0;
    const observer = new MutationObserver((records) => {
      for (const record of records) {
        for (const added of record.addedNodes) {
          if (added instanceof Element && added.closest('.react-flow__node')) {
            addedFieldInputs += added.matches('input') ? 1 : added.querySelectorAll('input').length;
          }
        }
      }
    });
    observer.observe(host, { childList: true, subtree: true });

    await renderEditor(root, createProjectStore(graph));

    const viewportKey = getWorkflowViewportKey(PROJECT_ID, graph.id, INSTANCE_ID);
    await expect.poll(() => getWorkflowViewport(viewportKey)).not.toBeNull();
    observer.disconnect();

    return { addedFieldInputs, zoom: getWorkflowViewport(viewportKey)!.zoom };
  };

  it('never mounts field controls for a graph whose fit lands zoomed out', async () => {
    const nodes = Array.from({ length: 6 }, (_, index) => addNode(`node-${index}`, index * 600, (index % 2) * 400));
    const fitted = await openFitted({ ...createProjectGraph('wide'), nodes });

    expect(fitted.zoom).toBeLessThan(CONTENT_VISIBILITY_ZOOM);
    expect(host.querySelectorAll('.react-flow__node')).toHaveLength(6);
    expect(fitted.addedFieldInputs).toBe(0);
  });

  it('mounts field controls for a graph that fits at a readable zoom', async () => {
    const nodes = [addNode('node-a', 0, 0), addNode('node-b', 0, 220)];
    const fitted = await openFitted({ ...createProjectGraph('compact'), nodes });

    expect(fitted.zoom).toBeGreaterThanOrEqual(CONTENT_VISIBILITY_ZOOM);
    expect(host.querySelectorAll('.react-flow__node input')).toHaveLength(2);
  });

  it('fits a graph clear of the floating toolbar, so the nodes beside it take their own clicks', async () => {
    const nodes = [addNode('left', 0, 0), addNode('right', 700, 0)];
    await openFitted({ ...createProjectGraph('beside-toolbar'), nodes });
    const toolbar = host.querySelector<HTMLElement>('[role="toolbar"]')!.getBoundingClientRect();
    const left = host.querySelector<HTMLElement>('.react-flow__node[data-id="left"]')!;
    const rect = left.getBoundingClientRect();

    // The fitted graph's left column sits level with the toolbar's column.
    expect(rect.top).toBeLessThan(toolbar.bottom);
    expect(rect.left).toBeGreaterThanOrEqual(toolbar.right);
    const hit = document.elementFromPoint(rect.left + 4, rect.top + rect.height / 2);
    expect(hit !== null && left.contains(hit)).toBe(true);
  });

  it('shows the enabled minimap only once the graph has a node', async () => {
    const empty = createProjectGraph('workflow-empty');
    const project = createProjectStore(empty);
    const minimap = () => host.querySelector('.react-flow__minimap');

    await renderEditor(root, project, { showMinimap: true });
    await expect.poll(getWorkflowFlowInstance).toBeTruthy();

    expect(minimap()).toBeNull();

    await act(() => project.setGraph({ ...empty, nodes: [addNode('only', 0, 0)] }));

    await expect.poll(minimap).not.toBeNull();
  });

  it('gives the next workflow its own flow, without errors from the outgoing one', async () => {
    const workflowA = {
      ...createProjectGraph('workflow-a'),
      nodes: Array.from({ length: 12 }, (_, index) => addNode(`a-${index}`, index * 300, 0)),
    };
    const workflowB = { ...createProjectGraph('workflow-b'), nodes: [addNode('b-0', 0, 0)] };
    const project = createProjectStore(workflowA);
    const nodeIds = () => [...host.querySelectorAll('.react-flow__node')].map((node) => node.getAttribute('data-id'));

    await renderEditor(root, project);
    await expect.poll(nodeIds).toHaveLength(12);
    // The flow registers its instance on init, which can land after the nodes are in the DOM.
    await expect.poll(getWorkflowFlowInstance).toBeTruthy();
    const outgoingFlow = getWorkflowFlowInstance();

    const consoleError = vi.spyOn(console, 'error');
    const consoleWarn = vi.spyOn(console, 'warn');
    const uncaught: unknown[] = [];
    const onError = (event: ErrorEvent) => uncaught.push(event.error);
    window.addEventListener('error', onError);

    try {
      await act(() => project.setGraph(workflowB));
      await expect.poll(nodeIds).toEqual(['b-0']);
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      });
    } finally {
      window.removeEventListener('error', onError);
    }

    // Unrelated act() notices from the fit's frame callbacks are not what this guards.
    const logged = [...consoleError.mock.calls, ...consoleWarn.mock.calls]
      .map((args) => String(args[0]))
      .filter((message) => !message.includes('not wrapped in act('));
    consoleError.mockRestore();
    consoleWarn.mockRestore();

    // Each workflow owns its flow store: a shared one notifies the outgoing nodes of the incoming graph while they
    // unmount, and each fails to find itself in it.
    expect(getWorkflowFlowInstance()).not.toBe(outgoingFlow);
    expect(outgoingFlow?.getNodes().map((node) => node.id)).not.toContain('b-0');
    expect(uncaught).toEqual([]);
    expect(logged).toEqual([]);
  });
});

describe('WorkflowEditorView fit selection', () => {
  let host: HTMLDivElement;
  let root: Root;
  const nodeIds = ['node-a', 'node-b', 'outside'];
  const graph = {
    ...createProjectGraph('fit-selection'),
    nodes: [addNode('node-a', 0, 0), addNode('node-b', 360, 0), addNode('outside', 1300, 0)],
  } satisfies ProjectGraphState;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.cssText = 'width: 480px; height: 520px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    clearWorkflowViewports();
  });

  const getNode = (id: string) => host.querySelector<HTMLElement>(`.react-flow__node[data-id="${id}"]`)!;
  const getFullyVisibleNodes = () => {
    const viewport = host.getBoundingClientRect();
    return nodeIds.filter((id) => {
      const node = getNode(id)?.getBoundingClientRect();
      return (
        node &&
        node.left >= viewport.left - 1 &&
        node.right <= viewport.right + 1 &&
        node.top >= viewport.top - 1 &&
        node.bottom <= viewport.bottom + 1
      );
    });
  };
  const getSelectedNodes = () =>
    getWorkflowFlowInstance()
      ?.getNodes()
      .filter((node) => node.selected)
      .map(({ id }) => id);

  it.each([
    ['the selection rectangle', ['node-a', 'node-b']],
    ['a selected node', ['node-a', 'node-b']],
    ['an unselected node', ['outside']],
  ])('fits the nodes targeted by the menu on %s without editing the graph', async (target, expectedNodes) => {
    const adapter = await renderEditor(root, createProjectStore(graph));
    await expect
      .poll(() => getWorkflowViewport(getWorkflowViewportKey(PROJECT_ID, graph.id, INSTANCE_ID)))
      .not.toBeNull();
    expect(getFullyVisibleNodes()).toEqual(nodeIds);

    await userEvent.click(host.querySelector<HTMLElement>('button[aria-label="widgets.workflow.toolbar.boxSelect"]')!);
    const pane = host.querySelector<HTMLElement>('.react-flow__pane')!;
    const paneRect = pane.getBoundingClientRect();
    const first = getNode('node-a').getBoundingClientRect();
    const last = getNode('node-b').getBoundingClientRect();
    await userEvent.dragAndDrop(pane, pane, {
      sourcePosition: { x: first.left - paneRect.left - 4, y: first.top - paneRect.top - 4 },
      targetPosition: { x: last.right - paneRect.left + 4, y: last.bottom - paneRect.top + 4 },
    });
    await expect.poll(getSelectedNodes).toEqual(['node-a', 'node-b']);

    const element =
      target === 'the selection rectangle'
        ? host.querySelector<HTMLElement>('.react-flow__nodesselection-rect')!
        : getNode(target === 'a selected node' ? 'node-a' : 'outside');
    const rect = element.getBoundingClientRect();
    await act(() =>
      element.dispatchEvent(
        new MouseEvent('contextmenu', {
          bubbles: true,
          cancelable: true,
          clientX: rect.left + 10,
          clientY: rect.top + 10,
        })
      )
    );
    const menuItem = document.querySelector<HTMLElement>('[role="menuitem"][data-value="fit-selection"]')!;
    await userEvent.click(menuItem);

    await expect.poll(getFullyVisibleNodes).toEqual(expectedNodes);
    expect(getSelectedNodes()).toEqual(['node-a', 'node-b']);
    expect(adapter.commands.editGraph).not.toHaveBeenCalled();

    await userEvent.click(host.querySelector<HTMLElement>('button[aria-label="widgets.workflow.toolbar.fitView"]')!);
    await expect.poll(getFullyVisibleNodes).toEqual(nodeIds);
  });
});

describe('WorkflowEditorView paste', () => {
  let host: HTMLDivElement;
  let outside: HTMLButtonElement;
  let root: Root;
  const graph = {
    ...createProjectGraph('paste'),
    edges: [
      {
        id: 'outside-to-a',
        source: 'outside',
        sourceHandle: 'value',
        target: 'node-a',
        targetHandle: 'a',
        type: 'default',
      },
    ],
    nodes: [addNode('node-a', 0, 0), addNode('node-b', 100, 50), addNode('outside', -300, 0)],
  } satisfies ProjectGraphState;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.cssText = 'width: 480px; height: 520px;';
    outside = document.createElement('button');
    outside.textContent = 'Side panel control';
    document.body.append(host, outside);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    outside.remove();
    clearWorkflowViewports();
    clearWorkflowClipboard();
  });

  /** Opens the editor with nodes A and B copied, and reports the pasted node positions of the next paste. */
  const openWithCopiedNodes = async (nodeIds = ['node-a', 'node-b']) => {
    const adapter = await renderEditor(root, createProjectStore(graph));
    await expect.poll(getWorkflowFlowInstance).toBeTruthy();
    await expect.poll(() => host.querySelectorAll('.react-flow__node').length).toBe(3);
    // The mount fit moves the view once; points are read in the view it settles on.
    await expect
      .poll(() => getWorkflowViewport(getWorkflowViewportKey(PROJECT_ID, graph.id, INSTANCE_ID)))
      .not.toBeNull();
    copyNodesToClipboard(graph, nodeIds);

    const pasted = () => {
      const action = vi
        .mocked(adapter.commands.editGraph)
        .mock.calls.map(([call]) => call)
        .find((call) => call.type === 'addGraphElements');

      return action?.type === 'addGraphElements' ? action : null;
    };

    return { pasted };
  };

  /** Moves the real pointer over an empty corner of the pane and returns where it landed. */
  const hoverPane = async (): Promise<{ x: number; y: number }> => {
    let pointer = { x: 0, y: 0 };
    const record = (event: PointerEvent) => {
      pointer = { x: event.clientX, y: event.clientY };
    };
    document.addEventListener('pointermove', record);
    await userEvent.hover(host.querySelector('.react-flow__pane')!, { position: { x: 440, y: 480 } });
    document.removeEventListener('pointermove', record);

    return pointer;
  };

  const viewCenter = () => {
    const rect = host.getBoundingClientRect();

    return getWorkflowFlowInstance()!.screenToFlowPosition({
      x: rect.left + rect.width / 2,
      y: rect.top + rect.height / 2,
    });
  };

  it("puts the group's top-left at the pointer when the editor has focus and the pointer is over it", async () => {
    const { pasted } = await openWithCopiedNodes();
    const pointer = await hoverPane();
    host.querySelector<HTMLElement>('.react-flow__node[data-id="node-a"]')!.focus();

    await act(() => commandHandlers.get('workflows.pasteSelection')!());

    const anchor = getWorkflowFlowInstance()!.screenToFlowPosition(pointer);
    expect(pasted()?.nodes.map((node) => node.position)).toEqual([
      { x: anchor.x, y: anchor.y },
      { x: anchor.x + 100, y: anchor.y + 50 },
    ]);
  });

  it('centers the group in the visible view when focus is outside the editor', async () => {
    const { pasted } = await openWithCopiedNodes();
    await hoverPane();
    outside.focus();

    await act(() => commandHandlers.get('workflows.pasteSelection')!());

    const center = viewCenter();
    expect(pasted()?.nodes.map((node) => node.position)).toEqual([
      { x: center.x - 50, y: center.y - 25 },
      { x: center.x + 50, y: center.y + 25 },
    ]);
  });

  it('centers the group when the pointer has left the editor, even with focus inside it', async () => {
    const { pasted } = await openWithCopiedNodes();
    await hoverPane();
    await userEvent.hover(outside);
    host.querySelector<HTMLElement>('.react-flow__node[data-id="node-a"]')!.focus();

    await act(() => commandHandlers.get('workflows.pasteSelection')!());

    const center = viewCenter();
    expect(pasted()?.nodes[0]?.position).toEqual({ x: center.x - 50, y: center.y - 25 });
  });

  /** Picks a context menu item the way a hand does, which also moves focus into the menu. */
  const chooseMenuItem = async (value: string) => {
    const item = await vi.waitFor(() => {
      const element = document.querySelector<HTMLElement>(`[role="menuitem"][data-value="${value}"]`);
      expect(element?.getAttribute('aria-disabled')).not.toBe('true');
      return element!;
    });
    await userEvent.click(item);
  };

  /** A node menu offers no paste point of its own: the group lands centered in the view, not where it was opened. */
  const expectCenteredPaste = (pasted: () => { nodes: { position: { x: number; y: number } }[] } | null) => {
    const center = viewCenter();
    expect(pasted()?.nodes.map((node) => node.position)).toEqual([
      { x: center.x - 50, y: center.y - 25 },
      { x: center.x + 50, y: center.y + 25 },
    ]);
  };

  it.each([false, true])(
    'centers a paste chosen from a node’s own menu (with connections: %s) instead of anchoring it at the node',
    async (withConnections) => {
      const { pasted } = await openWithCopiedNodes();
      const node = host.querySelector<HTMLElement>('.react-flow__node[data-id="node-b"]')!;
      const rect = node.getBoundingClientRect();

      await act(() =>
        node.dispatchEvent(
          new MouseEvent('contextmenu', {
            bubbles: true,
            cancelable: true,
            clientX: rect.left + 10,
            clientY: rect.top + 10,
          })
        )
      );
      await chooseMenuItem(withConnections ? 'paste-with-connections' : 'paste');

      expectCenteredPaste(pasted);
    }
  );

  it('centers a paste chosen from the selection’s menu instead of anchoring it where the menu opened', async () => {
    const { pasted } = await openWithCopiedNodes();
    const pane = host.querySelector<HTMLElement>('.react-flow__pane')!;
    const paneRect = pane.getBoundingClientRect();

    // Box-select every node, so the selection's own rectangle (and its menu) appears.
    await userEvent.click(host.querySelector<HTMLElement>('button[aria-label="widgets.workflow.toolbar.boxSelect"]')!);
    await userEvent.dragAndDrop(pane, pane, {
      sourcePosition: { x: paneRect.width - 4, y: 4 },
      targetPosition: { x: 60, y: paneRect.height - 4 },
    });
    const selection = await vi.waitFor(() => {
      const element = host.querySelector<HTMLElement>('.react-flow__nodesselection-rect');
      expect(element).not.toBeNull();
      return element!;
    });
    const rect = selection.getBoundingClientRect();

    await act(() =>
      selection.dispatchEvent(
        new MouseEvent('contextmenu', {
          bubbles: true,
          cancelable: true,
          clientX: rect.left + 10,
          clientY: rect.top + 10,
        })
      )
    );
    await chooseMenuItem('paste');

    expectCenteredPaste(pasted);
  });

  it('anchors a paste chosen from the empty canvas menu at the point the menu was opened on', async () => {
    const { pasted } = await openWithCopiedNodes();
    const pane = host.querySelector<HTMLElement>('.react-flow__pane')!;
    const paneRect = pane.getBoundingClientRect();
    const point = { x: paneRect.left + paneRect.width - 40, y: paneRect.top + paneRect.height - 40 };

    // Read before the menu opens: focusing it can scroll the test page, which moves every client point.
    const anchor = getWorkflowFlowInstance()!.screenToFlowPosition(point);

    await act(() =>
      pane.dispatchEvent(
        new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: point.x, clientY: point.y })
      )
    );
    await chooseMenuItem('paste');

    expect(pasted()?.nodes.map((node) => node.position)).toEqual([
      { x: anchor.x, y: anchor.y },
      { x: anchor.x + 100, y: anchor.y + 50 },
    ]);
  });

  it('keeps the copied node fed by its outside source only when pasting with connections', async () => {
    const { pasted } = await openWithCopiedNodes(['node-a']);

    await act(() => commandHandlers.get('workflows.pasteSelectionWithEdges')!());

    const action = pasted();
    expect(action?.edges).toEqual([expect.objectContaining({ source: 'outside', target: action?.nodes[0]?.id })]);
  });
});

describe('WorkflowEditorView edges and minimap', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.cssText = 'width: 480px; height: 520px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    clearWorkflowViewports();
  });

  const edge = (id: string, source: string, target: string) => ({
    id,
    source,
    sourceHandle: 'value',
    target,
    targetHandle: 'a',
    type: 'default' as const,
  });
  const collapsed = (node: WorkflowInvocationNode): WorkflowInvocationNode => ({
    ...node,
    data: { ...node.data, isOpen: false },
  });
  const edgeElement = (id: string) => host.querySelector<SVGGElement>(`.react-flow__edge[data-id="${id}"]`);
  const edgeOpacity = (id: string) =>
    getComputedStyle(edgeElement(id)!.querySelector<SVGPathElement>('.react-flow__edge-path')!).opacity;
  const editor = () => host.querySelector<HTMLElement>('.react-flow')!.parentElement!;

  it('keeps a selected node’s edges at full strength and dims the rest to a quarter', async () => {
    const graph = {
      ...createProjectGraph('emphasis'),
      edges: [edge('ab', 'a', 'b'), edge('cd', 'c', 'd')],
      nodes: [addNode('a', 0, 0), addNode('b', 400, 0), addNode('c', 0, 300), addNode('d', 400, 300)],
    };
    await renderEditor(root, createProjectStore(graph));
    await expect.poll(() => edgeElement('cd')).not.toBeNull();
    const restingOpacity = edgeOpacity('cd');

    await act(() => userEvent.click(host.querySelector<HTMLElement>('.react-flow__node[data-id="a"]')!));
    await expect.poll(() => editor().hasAttribute('data-node-selected')).toBe(true);
    expect(edgeElement('ab')!.classList.contains('workflow-selected-node-edge')).toBe(true);
    await expect.poll(() => edgeOpacity('cd')).toBe('0.25');
    await expect.poll(() => edgeOpacity('ab')).toBe('1');

    await act(() =>
      userEvent.click(host.querySelector<HTMLElement>('.react-flow__pane')!, { position: { x: 460, y: 500 } })
    );
    await expect.poll(() => editor().hasAttribute('data-node-selected')).toBe(false);
    await expect.poll(() => edgeOpacity('cd')).toBe(restingOpacity);
    expect(restingOpacity).not.toBe('0.25');
  });

  it('exports edges at the same strength whether or not a node is selected in the editor', async () => {
    const graph = {
      ...createProjectGraph('emphasis-export'),
      edges: [edge('ab', 'a', 'b'), edge('cd', 'c', 'd')],
      nodes: [addNode('a', 0, 0), addNode('b', 400, 0), addNode('c', 0, 300), addNode('d', 400, 300)],
    };
    await renderEditor(root, createProjectStore(graph));
    await expect.poll(() => edgeElement('cd')).not.toBeNull();
    const exportedOpacities = async () => {
      let opacities: string[] = [];
      exportMocks.rasterize.mockReset();
      exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
        opacities = [...clone.querySelectorAll<SVGPathElement>('.react-flow__edge .react-flow__edge-path')].map(
          (path) => getComputedStyle(path).opacity
        );
        return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
      });
      await act(() =>
        userEvent.click(host.querySelector<HTMLElement>('button[aria-label="widgets.workflow.exportAsPng"]')!)
      );
      await expect.poll(() => exportMocks.rasterize.mock.calls.length).toBe(1);
      await expect
        .poll(() => host.querySelector('button[aria-label="widgets.workflow.exportAsPng"]:disabled'))
        .toBeNull();
      return opacities;
    };

    const unselected = await exportedOpacities();
    expect(unselected).toHaveLength(2);
    await act(() => userEvent.click(host.querySelector<HTMLElement>('.react-flow__node[data-id="a"]')!));
    await expect.poll(() => editor().hasAttribute('data-node-selected')).toBe(true);

    expect(await exportedOpacities()).toEqual(unselected);
  });

  it('selects and deletes every edge a collapsed edge stands for, and draws them again when they return', async () => {
    const graph = {
      ...createProjectGraph('collapsed-edges'),
      edges: [edge('e1', 'a', 'b'), edge('e2', 'a', 'b')],
      nodes: [collapsed(addNode('a', 0, 0)), collapsed(addNode('b', 400, 160))],
    };
    const project = createProjectStore(graph);
    const adapter = await renderEditor(root, project);
    const drawn = () =>
      [...host.querySelectorAll('.react-flow__edge')].map((element) => element.getAttribute('data-id'));
    await expect.poll(drawn).toEqual(['e1']);
    const selectMergedEdge = async () => {
      await act(() =>
        userEvent.click(edgeElement('e1')!.querySelector('.react-flow__edge-interaction')!, { force: true })
      );
      await expect.poll(() => edgeElement('e1')!.classList.contains('selected')).toBe(true);
    };

    // Its ends cannot be dragged: a reconnection would move one of the edges it stands for.
    expect(edgeElement('e1')!.querySelector('.react-flow__edgeupdater')).toBeNull();

    // The Delete command (palette or hotkey) and the Delete key delete the same set.
    await selectMergedEdge();
    await act(() => commandHandlers.get('workflows.deleteSelection')!());
    expect(adapter.commands.editGraph).toHaveBeenCalledWith({ edgeIds: ['e1', 'e2'], type: 'removeEdges' });
    vi.mocked(adapter.commands.editGraph).mockClear();
    await act(() => userEvent.keyboard('{Delete}'));
    expect(adapter.commands.editGraph).toHaveBeenCalledWith({ edgeIds: ['e1', 'e2'], type: 'removeEdges' });

    // Undo hands the editor the same edges again: they draw as one collapsed edge with its count.
    await act(() => project.setGraph({ ...graph, edges: [] }));
    await expect.poll(drawn).toEqual([]);
    await act(() => project.setGraph(graph));
    await expect.poll(drawn).toEqual(['e1']);
    expect(host.querySelector('[data-collapsed-edge-count]')?.getAttribute('data-collapsed-edge-count')).toBe('2');
  });

  it('keeps a merged line selected when it is clicked again, so Delete still removes every edge it stands for', async () => {
    const graph = {
      ...createProjectGraph('collapsed-reclick'),
      edges: [edge('e1', 'a', 'b'), edge('e2', 'a', 'b')],
      nodes: [collapsed(addNode('a', 0, 0)), collapsed(addNode('b', 400, 160))],
    };
    const adapter = await renderEditor(root, createProjectStore(graph));
    await expect.poll(() => edgeElement('e1')).not.toBeNull();
    const clickMergedLine = () =>
      act(() => userEvent.click(edgeElement('e1')!.querySelector('.react-flow__edge-interaction')!, { force: true }));

    await clickMergedLine();
    await clickMergedLine();
    await expect.poll(() => edgeElement('e1')!.classList.contains('selected')).toBe(true);
    await act(() => userEvent.keyboard('{Delete}'));

    expect(adapter.commands.editGraph).toHaveBeenCalledWith({ edgeIds: ['e1', 'e2'], type: 'removeEdges' });
  });

  it('leaves a merged line’s edges agreeing after a box selection, so deleting takes all or none of them', async () => {
    const graph = {
      ...createProjectGraph('collapsed-box'),
      edges: [edge('e1', 'a', 'b'), edge('e2', 'a', 'b'), edge('cd', 'c', 'd')],
      nodes: [
        collapsed(addNode('a', 0, 0)),
        collapsed(addNode('b', 400, 160)),
        addNode('c', 0, 320),
        addNode('d', 400, 480),
      ],
    };
    const adapter = await renderEditor(root, createProjectStore(graph));
    await expect.poll(() => edgeElement('cd')).not.toBeNull();
    const flowRect = host.querySelector<HTMLElement>('.react-flow')!.getBoundingClientRect();

    // Shift-drag across the whole graph box-selects every node. (The selection's own rectangle then covers the
    // merged line, so it takes no click until the selection is cleared; the click case is in flowAdapters.test.)
    await userEvent.keyboard('{Shift>}');
    await commands.mouseDragPath(
      [
        { x: flowRect.left + 2, y: flowRect.top + 2 },
        { x: flowRect.left + 40, y: flowRect.top + 40 },
        { x: flowRect.right - 2, y: flowRect.bottom - 2 },
      ],
      window.innerWidth
    );
    await userEvent.keyboard('{/Shift}');
    await expect
      .poll(() => host.querySelector('.react-flow__node[data-id="a"]')!.classList.contains('selected'))
      .toBe(true);

    await act(() => commandHandlers.get('workflows.deleteSelection')!());
    const removedEdges = vi
      .mocked(adapter.commands.editGraph)
      .mock.calls.flatMap(([action]) => (action.type === 'removeEdges' ? action.edgeIds : []));
    expect(removedEdges.includes('e1')).toBe(removedEdges.includes('e2'));
  });

  it('lets only the open end of an edge from a collapsed node be dragged', async () => {
    const graph = {
      ...createProjectGraph('collapsed-to-open'),
      edges: [edge('ac', 'a', 'c')],
      nodes: [collapsed(addNode('a', 0, 0)), addNode('c', 400, 160)],
    };
    await renderEditor(root, createProjectStore(graph));
    await expect.poll(() => edgeElement('ac')).not.toBeNull();

    expect(edgeElement('ac')!.querySelector('.react-flow__edgeupdater-source')).toBeNull();
    expect(edgeElement('ac')!.querySelector('.react-flow__edgeupdater-target')).not.toBeNull();
  });

  it('tints invalid and failed nodes in the error tone and the running node in the accent tone on the minimap', async () => {
    const graph = {
      ...createProjectGraph('minimap-states'),
      nodes: [
        addNode('idle', 0, 0),
        { ...addNode('unknown', 300, 0), data: { ...addNode('unknown', 300, 0).data, type: 'not_installed' } },
        addNode('running', 0, 300),
        addNode('failed', 300, 300),
      ],
    };
    const runState = (status: 'failed' | 'running') => ({
      error: null,
      latestOutput: null,
      outputImageName: null,
      progress: null,
      progressMessage: null,
      status,
    });
    // Stable snapshots, as the real port keeps them between changes.
    const origin = { projectId: PROJECT_ID, workflowId: graph.id };
    const states: Record<string, ReturnType<typeof runState>> = {
      failed: runState('failed'),
      running: runState('running'),
    };
    const subscribe = () => () => {};
    await renderEditor(root, createProjectStore(graph), {
      nodeExecution: {
        get: (nodeId) => states[nodeId] ?? null,
        getOrigin: () => origin,
        subscribe,
        subscribeOrigin: subscribe,
      },
      showMinimap: true,
    });

    const minimapNodes = () => [...host.querySelectorAll<SVGRectElement>('.react-flow__minimap-node')];
    await expect.poll(() => minimapNodes().length).toBe(4);
    const classes = minimapNodes().map((rect) =>
      rect.classList.contains('workflow-minimap-error')
        ? 'error'
        : rect.classList.contains('workflow-minimap-running')
          ? 'running'
          : 'plain'
    );
    expect(classes).toEqual(['plain', 'error', 'running', 'error']);
    const fills = minimapNodes().map((rect) => getComputedStyle(rect).fill);
    expect(fills[1]).not.toBe(fills[0]);
    expect(fills[2]).not.toBe(fills[0]);
    expect(fills[2]).not.toBe(fills[1]);
    expect(fills[3]).toBe(fills[1]);
  });
});
