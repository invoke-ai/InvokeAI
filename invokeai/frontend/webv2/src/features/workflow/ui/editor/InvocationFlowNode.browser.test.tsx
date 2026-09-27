import type { InvocationTemplate, ProjectGraphState, WorkflowInvocationNode } from '@features/workflow/contracts';
import type { WorkflowNodeExecutionState } from '@features/workflow/ui/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';
import type { ProjectGraphAction } from '@features/workflow/utility';

/* eslint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop -- each render mounts a fresh flow on purpose */
import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { setNodePreviewCollapsed } from '@features/workflow/ui/workflowUiStore';
import { createProjectGraph, projectGraphReducer } from '@features/workflow/utility';
import { system } from '@theme/system';
import { applyNodeChanges, ReactFlow, type NodeChange } from '@xyflow/react';
import { act, startTransition, useCallback, useEffect, useState, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { toFlowEdges, toFlowNodes } from './flowAdapters';
import { InvocationFlowNode } from './InvocationFlowNode';

import '@xyflow/react/dist/style.css';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { error?: string }) =>
      ({
        'nodes.childWorkflowError': `Child workflow error: ${options?.error ?? ''}`,
        'nodes.executionFailed': 'Failed',
        'nodes.executionCompleted': 'Completed',
        'nodes.latestOutput': 'Latest output',
        'nodes.latestOutputImage': 'Latest output of this node',
      })[key] ?? key,
  }),
}));

const NODE_ID = 'preview-node';
/** The preview's fixed height (10rem) in CSS pixels, however the root font is sized. */
const previewHeightPx = () => parseFloat(getComputedStyle(document.documentElement).fontSize) * 10;

const template: InvocationTemplate = {
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: {
    a: {
      default: undefined,
      description: '',
      exclusiveMaximum: null,
      exclusiveMinimum: null,
      fieldKind: 'input',
      input: 'connection',
      maximum: null,
      minimum: null,
      multipleOf: null,
      name: 'a',
      options: null,
      required: false,
      title: 'A',
      type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
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
  outputType: 'test',
  outputs: {
    value: {
      description: '',
      name: 'value',
      title: 'Value',
      type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
    },
  },
  tags: [],
  title: 'Preview',
  type: 'preview',
  useCache: true,
  version: '1.0.0',
};

const documentNode: WorkflowInvocationNode = {
  data: {
    inputs: {},
    isIntermediate: true,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type: 'preview',
    useCache: true,
    version: '1.0.0',
  },
  id: NODE_ID,
  position: { x: 20, y: 20 },
  type: 'invocation',
};

const projectGraph: ProjectGraphState = { ...createProjectGraph('preview-test'), nodes: [documentNode] };
const templates = { preview: template };
const flowNodes = toFlowNodes(projectGraph, [], templates);
const nodeTypes = { invocation: InvocationFlowNode };

const callNodeId = 'call-node';
const callNode: WorkflowInvocationNode = {
  ...documentNode,
  data: { ...documentNode.data, type: 'call_saved_workflow' },
  id: callNodeId,
};
const callTemplate: InvocationTemplate = {
  ...template,
  inputs: {},
  outputType: 'workflow_return_output',
  outputs: {},
  title: 'Call Saved Workflow',
  type: 'call_saved_workflow',
};
const callProjectGraph: ProjectGraphState = { ...createProjectGraph('call-test'), nodes: [callNode] };
const callFlowNodes = toFlowNodes(callProjectGraph, [], { call_saved_workflow: callTemplate });

const outputImage = (width: number, height: number): string => {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d')!;
  context.fillStyle = '#4c8bf5';
  context.fillRect(0, 0, width, height);
  return canvas.toDataURL();
};

const completed = (outputImageUrl: string): WorkflowNodeExecutionState => ({
  error: null,
  latestOutput: null,
  outputImageUrl,
  progress: null,
  progressMessage: null,
  status: 'completed',
});

/** A node-execution port the test can advance between renders. */
const createExecutionPort = (nodeId = NODE_ID) => {
  const listeners = new Set<() => void>();
  let state: WorkflowNodeExecutionState | null = null;

  return {
    port: {
      get: (requestedNodeId: string) => (requestedNodeId === nodeId ? state : null),
      subscribe: (_nodeId: string, listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    },
    set(next: WorkflowNodeExecutionState) {
      state = next;
      listeners.forEach((listener) => listener());
    },
  };
};

const preferencesSnapshot = {
  reduceMotion: true,
  themeId: 'classic' as const,
  workflowEdgeStyle: 'curved' as const,
  workflowEdgesBehindNodes: false,
  workflowShowMinimap: false,
  workflowSnapToGrid: false,
  workflowValidateConnections: true,
};
const projectSnapshot = {
  galleryValues: {},
  id: 'project-1',
  isWorkflowRunning: false,
  projectGraph,
  workflowValues: {},
};

const createAdapter = (nodeExecution: WorkflowUiAdapter['nodeExecution']): WorkflowUiAdapter =>
  ({
    capabilities: { getSnapshot: () => ({ canUseCache: true }), subscribe: () => () => {} },
    commands: { bindLibraryWorkflow: vi.fn(), editGraph: vi.fn(), redo: vi.fn(), replace: vi.fn(), undo: vi.fn() },
    getProjectGraph: () => projectGraph,
    nodeExecution,
    notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
    openAddModels: vi.fn(),
    performance: {
      mark: vi.fn(),
      measure: vi.fn(),
      time: <T,>(_name: string, _source: unknown, callback: () => T) => callback(),
    },
    preferences: { getSnapshot: () => preferencesSnapshot, subscribe: () => () => {} },
    project: { getSnapshot: () => projectSnapshot, subscribe: () => () => {} },
    registerModalHotkeyLayer: vi.fn(() => vi.fn()),
    widgets: { open: vi.fn(), patchValues: vi.fn() },
  }) as unknown as WorkflowUiAdapter;

describe('InvocationFlowNode output preview', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    setNodePreviewCollapsed(NODE_ID, false);
    host = document.createElement('div');
    host.style.cssText = 'width: 480px; height: 520px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  const render = (adapter: WorkflowUiAdapter, zoom = 1) =>
    act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <ReactFlow
              defaultViewport={{ x: 0, y: 0, zoom }}
              edges={[]}
              minZoom={0.1}
              nodes={flowNodes}
              nodeTypes={nodeTypes}
            />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );
  const image = () => host.querySelector<HTMLImageElement>('.react-flow__node img');
  const disclosure = () =>
    [...host.querySelectorAll<HTMLButtonElement>('.react-flow__node button[aria-expanded]')].find(
      (button) => button.textContent === 'Latest output'
    )!;
  const nodeHeight = () => host.querySelector<HTMLElement>('.react-flow__node')!.getBoundingClientRect().height;

  it('keeps one preview height across differently shaped outputs and folds it away per node', async () => {
    const execution = createExecutionPort();
    const adapter = createAdapter(execution.port);

    await render(adapter);
    expect(image()).toBeNull();
    expect(disclosure()).toBeUndefined();

    await act(() => execution.set(completed(outputImage(400, 100))));
    await vi.waitFor(() => expect(image()!.getBoundingClientRect().height).toBeCloseTo(previewHeightPx(), 0));
    const heightWithWideOutput = nodeHeight();
    await page.screenshot({ path: '../../../../../artifacts/workflow-node-preview/expanded.png' });

    await act(() => execution.set(completed(outputImage(100, 400))));
    await vi.waitFor(() => expect(image()!.src).toBe(outputImage(100, 400)));
    expect(image()!.getBoundingClientRect().height).toBeCloseTo(previewHeightPx(), 0);
    expect(nodeHeight()).toBe(heightWithWideOutput);

    await act(() => disclosure().click());
    expect(disclosure().getAttribute('aria-expanded')).toBe('false');
    expect(image()).toBeNull();
    expect(nodeHeight()).toBeLessThan(heightWithWideOutput - 100);
    await page.screenshot({ path: '../../../../../artifacts/workflow-node-preview/collapsed.png' });

    // The fold outlives the node's mount: a remount (offscreen virtualization, a project switch back) keeps it.
    await act(() => root.unmount());
    root = createRoot(host);
    await render(adapter);
    expect(disclosure().getAttribute('aria-expanded')).toBe('false');
    expect(image()).toBeNull();

    await act(() => disclosure().click());
    expect(disclosure().getAttribute('aria-expanded')).toBe('true');
    await vi.waitFor(() => expect(image()!.getBoundingClientRect().height).toBeCloseTo(previewHeightPx(), 0));
  });

  it('stands in a same-height skeleton for the image when the viewport is zoomed out', async () => {
    const execution = createExecutionPort();
    const adapter = createAdapter(execution.port);
    execution.set(completed(outputImage(400, 100)));

    await render(adapter);
    await vi.waitFor(() => expect(image()).not.toBeNull());
    const node = () => host.querySelector<HTMLElement>('.react-flow__node')!;
    const layoutHeight = node().offsetHeight;

    await act(() => root.unmount());
    root = createRoot(host);
    await render(adapter, 0.3);

    await vi.waitFor(() => expect(image()).toBeNull());
    expect(node().offsetHeight).toBe(layoutHeight);
  });
});

describe('InvocationFlowNode failure outcome', () => {
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
  });

  it('shows translated child-workflow attribution in the failure tooltip', async () => {
    const execution = createExecutionPort(callNodeId);
    execution.set({
      error: 'Child node failed',
      latestOutput: null,
      outputImageUrl: null,
      progress: null,
      progressMessage: null,
      status: 'failed',
    });

    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={createAdapter(execution.port)}>
            <ReactFlow defaultViewport={{ x: 0, y: 0, zoom: 1 }} nodes={callFlowNodes} nodeTypes={nodeTypes} />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );

    const failureIcon = await vi.waitFor(() => {
      const icon = host.querySelector<HTMLElement>('[aria-label="Failed"]');
      expect(icon).not.toBeNull();
      return icon!;
    });

    await act(() => userEvent.hover(failureIcon));
    await vi.waitFor(() =>
      expect(document.querySelector('[data-part="content"]')?.textContent).toContain(
        'Child workflow error: Child node failed'
      )
    );
  });
});

describe('InvocationFlowNode edge stacking', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.cssText = 'width: 400px; height: 500px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  /** Screen-space samples along an edge's hit path that fall inside `rect`, with what the pointer would hit there. */
  const hitsInside = (path: SVGGeometryElement, rect: DOMRect) => {
    const svg = path.ownerSVGElement!;
    const matrix = svg.getScreenCTM()!;
    const length = path.getTotalLength();
    const hits: Element[] = [];

    for (let step = 0; step <= 80; step += 1) {
      const point = path.getPointAtLength((length * step) / 80).matrixTransform(matrix);

      if (point.x > rect.left + 2 && point.x < rect.right - 2 && point.y > rect.top + 2 && point.y < rect.bottom - 2) {
        hits.push(document.elementFromPoint(point.x, point.y)!);
      }
    }

    return hits;
  };

  it('keeps a selected node clickable where its own edge crosses it, while the edge stays above other nodes', async () => {
    // Route the edge back across its source and another node to test overlapping interaction paths within the
    // viewport.
    const target: WorkflowInvocationNode = { ...documentNode, id: 'target-node', position: { x: 0, y: 0 } };
    const bystander: WorkflowInvocationNode = { ...documentNode, id: 'bystander-node', position: { x: 20, y: 120 } };
    const graph: ProjectGraphState = {
      ...projectGraph,
      edges: [
        { id: 'e', source: NODE_ID, sourceHandle: 'value', target: target.id, targetHandle: 'a', type: 'default' },
      ],
      nodes: [{ ...documentNode, position: { x: 100, y: 220 } }, target, bystander],
    };
    const nodes = toFlowNodes(graph, [], templates).map((node) => ({ ...node, selected: node.id === NODE_ID }));
    const edges = toFlowEdges(graph, [], 'default', new Set([NODE_ID]), templates);

    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={createAdapter({ get: () => null, subscribe: () => () => {} })}>
            {/* The editor passes `elevateEdgesOnSelect` unless edges are kept behind nodes; xyflow defaults it off. */}
            <ReactFlow edges={edges} elevateEdgesOnSelect nodes={nodes} nodeTypes={nodeTypes} />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );

    const path = await vi.waitFor(() => {
      const element = host.querySelector<SVGGeometryElement>('.react-flow__edge-interaction');
      expect(element?.getTotalLength()).toBeGreaterThan(0);
      return element!;
    });
    const selectedNode = host.querySelector<HTMLElement>(`.react-flow__node[data-id="${NODE_ID}"]`)!;
    const otherNode = host.querySelector<HTMLElement>(`.react-flow__node[data-id="${bystander.id}"]`)!;
    const overSelected = hitsInside(path, selectedNode.getBoundingClientRect());
    const overOther = hitsInside(path, otherNode.getBoundingClientRect());

    expect(overSelected.length).toBeGreaterThan(0);
    expect(overSelected.every((hit) => selectedNode.contains(hit))).toBe(true);
    expect(overOther.length).toBeGreaterThan(0);
    expect(overOther.some((hit) => hit.closest('.react-flow__edge') !== null)).toBe(true);
  });
});

describe('InvocationFlowNode field entry', () => {
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
  });

  const entryInput = (name: string, title: string, typeName: string) => ({
    ...template.inputs.a!,
    input: 'any' as const,
    name,
    required: true,
    title,
    type: { batch: false, cardinality: 'SINGLE' as const, name: typeName },
  });
  const entryTemplate: InvocationTemplate = {
    ...template,
    inputs: {
      prompt: entryInput('prompt', 'Prompt', 'StringField'),
      steps: entryInput('steps', 'Steps', 'IntegerField'),
      weight: entryInput('weight', 'Weight', 'FloatField'),
    },
    outputs: {},
    title: 'Entry',
    type: 'entry',
  };
  const entryTemplates = { entry: entryTemplate };
  const entryNode: WorkflowInvocationNode = {
    ...documentNode,
    data: {
      ...documentNode.data,
      inputs: {
        prompt: { label: '', name: 'prompt', value: 'hello world' },
        steps: { label: '', name: 'steps', value: 20 },
        weight: { label: '', name: 'weight', value: 0.5 },
      },
      type: 'entry',
    },
    id: 'entry-node',
  };

  /** An undoable graph store standing in for the project aggregate. */
  const createGraphStore = (initial: ProjectGraphState) => {
    const listeners = new Set<() => void>();
    const history: ProjectGraphState[] = [];
    let graph = initial;
    const publish = (next: ProjectGraphState) => {
      graph = next;
      listeners.forEach((listener) => listener());
    };

    return {
      editGraph: (action: ProjectGraphAction) => {
        history.push(graph);
        publish(projectGraphReducer(graph, action));
      },
      getSnapshot: () => graph,
      subscribe: (listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      undo: () => {
        const previous = history.pop();
        if (previous) {
          publish(previous);
        }
      },
    };
  };
  type GraphStore = ReturnType<typeof createGraphStore>;

  const fieldValue = (store: GraphStore, name: string) => {
    const node = store.getSnapshot().nodes.find((candidate) => candidate.id === entryNode.id);
    return node?.type === 'invocation' ? node.data.inputs[name]?.value : undefined;
  };

  /**
   * Mirrors WorkflowEditorView: the flow model is rebuilt from the graph in a transition after an effect, so a
   * committed value echoes back to the node a beat after the keystroke that produced it. xyflow's node changes
   * (measured sizes) are applied as the editor does; a rebuilt node without them is hidden until remeasured and
   * drops the next keystroke.
   */
  const DeferredFlow = ({ store }: { store: GraphStore }) => {
    const graph = useSyncExternalStore(store.subscribe, store.getSnapshot);
    const [flowNodes, setFlowNodes] = useState(() => toFlowNodes(graph, [], entryTemplates));
    const onNodesChange = useCallback(
      (changes: NodeChange<(typeof flowNodes)[number]>[]) =>
        setFlowNodes((current) => applyNodeChanges(changes, current)),
      []
    );

    useEffect(() => {
      startTransition(() => {
        setFlowNodes((current) => toFlowNodes(graph, current, entryTemplates));
      });
    }, [graph]);

    return <ReactFlow edges={[]} nodes={flowNodes} nodeTypes={nodeTypes} onNodesChange={onNodesChange} />;
  };

  const renderEntryNode = async () => {
    const store = createGraphStore({ ...createProjectGraph('entry-test'), nodes: [entryNode] });
    const adapter = {
      ...createAdapter({ get: () => null, subscribe: () => () => {} }),
      commands: {
        bindLibraryWorkflow: vi.fn(),
        editGraph: store.editGraph,
        redo: vi.fn(),
        replace: vi.fn(),
        undo: store.undo,
      },
      getProjectGraph: store.getSnapshot,
      project: {
        getSnapshot: () => ({ ...projectSnapshot, projectGraph: store.getSnapshot() }),
        subscribe: store.subscribe,
      },
    } as unknown as WorkflowUiAdapter;

    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <DeferredFlow store={store} />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );

    const field = (label: string) =>
      host.querySelector<HTMLInputElement>(`.react-flow__node input[aria-label="${label}"]`)!;

    return { field, store };
  };
  const focusAtEnd = (input: HTMLInputElement) =>
    act(async () => {
      await userEvent.click(input);
      await userEvent.keyboard('{End}');
    });
  const keys = (sequence: string) =>
    act(async () => {
      await userEvent.keyboard(sequence);
    });
  /** Lets the deferred flow-model rebuild land so the assertion sees the echoed value, not just the draft. */
  const settle = () =>
    act(async () => {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  const rowError = (input: HTMLInputElement) =>
    input.closest('[data-scope="field"][data-part="root"]')?.querySelector('[data-part="error-text"]')?.textContent ??
    null;

  it('keeps focus and caret in a text field while the deferred flow model catches up', async () => {
    const { field, store } = await renderEntryNode();
    const prompt = field('Prompt');

    await focusAtEnd(prompt);
    await keys('{Home}{ArrowRight}{ArrowRight}XYZ');
    expect(prompt.value).toBe('heXYZllo world');
    expect([prompt.selectionStart, prompt.selectionEnd]).toEqual([5, 5]);

    await settle();
    expect(document.activeElement).toBe(prompt);
    expect(field('Prompt')).toBe(prompt);
    expect(prompt.value).toBe('heXYZllo world');
    expect([prompt.selectionStart, prompt.selectionEnd]).toEqual([5, 5]);
    expect(fieldValue(store, 'prompt')).toBe('heXYZllo world');

    await keys('{Shift>}{End}{/Shift}there');
    expect(prompt.value).toBe('heXYZthere');
    expect([prompt.selectionStart, prompt.selectionEnd]).toEqual([10, 10]);
    await keys('{Control>}a{/Control}{Backspace}');
    await settle();
    expect(prompt.value).toBe('');
    expect(fieldValue(store, 'prompt')).toBe('');
    expect(document.activeElement).toBe(prompt);
    expect(rowError(prompt)).toBeNull();

    // Undo from elsewhere (a toolbar button has focus) restores the text without pulling focus into a field.
    const weight = field('Weight');
    await act(async () => {
      await userEvent.click(host);
    });
    await act(() => store.undo());
    await settle();
    expect(prompt.value).toBe('heXYZthere');
    expect(document.activeElement).not.toBe(prompt);
    expect(document.activeElement).not.toBe(weight);
  });

  it('edits a float field in place and reports drafts, clears, and a leading minus through the graph', async () => {
    const { field, store } = await renderEntryNode();
    const weight = field('Weight');

    await focusAtEnd(weight);
    await keys('{Home}2');
    expect(weight.value).toBe('20.5');
    await settle();
    await keys('3');
    expect(weight.value).toBe('230.5');
    expect(fieldValue(store, 'weight')).toBe(230.5);
    expect(document.activeElement).toBe(weight);

    await keys('{Control>}a{/Control}0.60');
    await settle();
    expect(weight.value).toBe('0.60');
    expect(fieldValue(store, 'weight')).toBe(0.6);

    await keys('{Control>}a{/Control}2');
    await settle();
    expect(weight.value).toBe('2');
    expect(fieldValue(store, 'weight')).toBe(2);

    await keys('{Backspace}');
    await settle();
    expect(weight.value).toBe('');
    expect(fieldValue(store, 'weight')).toBeUndefined();
    expect(document.activeElement).toBe(weight);
    expect(rowError(weight)).toBe('Required value.');

    await keys('7{Home}-');
    await settle();
    expect(weight.value).toBe('-7');
    expect(fieldValue(store, 'weight')).toBe(-7);
    expect(rowError(weight)).toBeNull();

    await keys('{Home}{Delete}');
    await settle();
    expect(weight.value).toBe('7');
    expect(fieldValue(store, 'weight')).toBe(7);

    await act(() => weight.blur());
    await settle();
    expect(weight.value).toBe('7');
  });

  it('keeps a fractional integer visible and flagged until it is corrected', async () => {
    const { field, store } = await renderEntryNode();
    const steps = field('Steps');

    await focusAtEnd(steps);
    await keys('.5');
    await settle();
    expect(steps.value).toBe('20.5');
    expect(fieldValue(store, 'steps')).toBe(20.5);
    expect(steps.getAttribute('aria-invalid')).toBe('true');
    expect(rowError(steps)).toBe('Invalid value.');

    await keys('{Backspace}{Backspace}');
    await settle();
    expect(steps.value).toBe('20');
    expect(fieldValue(store, 'steps')).toBe(20);
    expect(steps.getAttribute('aria-invalid')).toBeNull();
    expect(rowError(steps)).toBeNull();
  });
});

describe('InvocationFlowNode template version', () => {
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
  });

  const render = (nodes: ReturnType<typeof toFlowNodes>) =>
    act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={createAdapter(createExecutionPort().port)}>
            <ReactFlow edges={[]} nodes={nodes} nodeTypes={nodeTypes} />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );
  const updateIcon = () => host.querySelector('.react-flow__node [role="img"][aria-label^="nodes.node"]');
  const borderColor = () => getComputedStyle(host.querySelector<HTMLElement>('.react-flow__node > div')!).borderColor;

  it('marks a node whose template moved on with a warning border and an update tooltip, and leaves a current one alone', async () => {
    await render(flowNodes);
    expect(updateIcon()).toBeNull();
    const currentBorder = borderColor();

    await render(toFlowNodes(projectGraph, [], { preview: { ...template, version: '1.1.0' } }));
    expect(updateIcon()?.getAttribute('aria-label')).toBe('nodes.nodeUpdateAvailable');
    expect(borderColor()).not.toBe(currentBorder);

    await render(toFlowNodes(projectGraph, [], { preview: { ...template, version: '2.0.0' } }));
    expect(updateIcon()?.getAttribute('aria-label')).toBe('nodes.nodeVersionIncompatible');
  });
});

describe('InvocationFlowNode batch nodes', () => {
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
  });

  const batchTemplate: InvocationTemplate = {
    ...template,
    inputs: {
      batch_group_id: {
        ...(template.inputs.a as InvocationTemplate['inputs'][string]),
        default: 'None',
        input: 'direct',
        name: 'batch_group_id',
        options: ['None', 'Group 1', 'Group 2'],
        title: 'Batch Group',
        type: { batch: false, cardinality: 'SINGLE', name: 'EnumField' },
      },
      floats: {
        ...(template.inputs.a as InvocationTemplate['inputs'][string]),
        default: [],
        input: 'any',
        name: 'floats',
        title: 'Floats',
        type: { batch: true, cardinality: 'COLLECTION', name: 'FloatField' },
      },
    },
    outputs: {
      value: {
        description: '',
        name: 'value',
        title: 'Value',
        type: { batch: false, cardinality: 'SINGLE', name: 'FloatField' },
      },
    },
    type: 'float_batch',
  };
  const batchNode = (groupId: string): WorkflowInvocationNode => ({
    ...documentNode,
    data: {
      ...documentNode.data,
      inputs: {
        batch_group_id: { label: '', name: 'batch_group_id', value: groupId },
        floats: { label: '', name: 'floats', value: [1, 2] },
      },
      type: 'float_batch',
    },
    id: 'batch-node',
  });
  const render = (groupId: string) => {
    const graph: ProjectGraphState = { ...createProjectGraph('batch-test'), nodes: [batchNode(groupId)] };

    return act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={createAdapter(createExecutionPort().port)}>
            <ReactFlow
              edges={[]}
              nodes={toFlowNodes(graph, [], { float_batch: batchTemplate })}
              nodeTypes={nodeTypes}
            />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );
  };
  const header = () => host.querySelector<HTMLElement>('.react-flow__node > div > div')!;

  it('names the group beside the title, draws the batch list handle as a diamond, and keeps the footer off', async () => {
    await render('Group 2');
    expect(header().textContent).toContain('(Group 2)');
    expect(host.querySelector<HTMLElement>('.react-flow__handle[data-handleid="floats"]')?.style.transform).toContain(
      'rotate(45deg)'
    );
    expect(host.textContent).not.toContain('Use Cache');

    await render('None');
    expect(header().textContent).toContain('(nodes.noBatchGroup)');
  });
});
