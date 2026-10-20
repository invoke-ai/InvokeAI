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
import { userEvent } from 'vitest/browser';
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

const createAdapter = (project: ReturnType<typeof createProjectStore>): WorkflowUiAdapter => {
  const subscribe = () => () => {};

  return {
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
    nodeExecution: { get: () => null, getOrigin: () => null, subscribe, subscribeOrigin: subscribe },
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
        workflowShowMinimap: false,
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
const renderEditor = async (root: Root, project: ReturnType<typeof createProjectStore>) => {
  const adapter = createAdapter(project);

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

  it('keeps the copied node fed by its outside source only when pasting with connections', async () => {
    const { pasted } = await openWithCopiedNodes(['node-a']);

    await act(() => commandHandlers.get('workflows.pasteSelectionWithEdges')!());

    const action = pasted();
    expect(action?.edges).toEqual([expect.objectContaining({ source: 'outside', target: action?.nodes[0]?.id })]);
  });
});
