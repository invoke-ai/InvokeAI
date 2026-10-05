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
import '@xyflow/react/dist/style.css';

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

const runtime = {
  commands: { register: () => () => {} },
  hotkeys: { register: () => () => {} },
  instanceId: INSTANCE_ID,
  region: 'center',
  typeId: 'workflow',
} as const;

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

  const renderEditor = (project: ReturnType<typeof createProjectStore>) =>
    act(() =>
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <QueryClientProvider client={new QueryClient()}>
              <WorkflowUiProvider adapter={createAdapter(project)}>
                <WorkflowEditorView runtime={runtime} />
              </WorkflowUiProvider>
            </QueryClientProvider>
          </ChakraProvider>
        </I18nextProvider>
      )
    );

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

    await renderEditor(createProjectStore(graph));

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

  it('gives the next workflow its own flow, without errors from the outgoing one', async () => {
    const workflowA = {
      ...createProjectGraph('workflow-a'),
      nodes: Array.from({ length: 12 }, (_, index) => addNode(`a-${index}`, index * 300, 0)),
    };
    const workflowB = { ...createProjectGraph('workflow-b'), nodes: [addNode('b-0', 0, 0)] };
    const project = createProjectStore(workflowA);
    const nodeIds = () => [...host.querySelectorAll('.react-flow__node')].map((node) => node.getAttribute('data-id'));

    await renderEditor(project);
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
