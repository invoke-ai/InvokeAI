import type { InvocationTemplate, ProjectGraphState, WorkflowInvocationNode } from '@features/workflow/contracts';
import type { WorkflowNodeExecutionState } from '@features/workflow/ui/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { workflowSelectionStore } from '@features/workflow/ui/editor/selectionStore';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { NodeInspector } from './NodeInspector';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const template: InvocationTemplate = {
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: {},
  nodePack: 'invokeai',
  outputType: 'test',
  outputs: {
    height: {
      description: '',
      name: 'height',
      title: 'Height',
      type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
    },
    width: {
      description: '',
      name: 'width',
      title: 'Width',
      type: { batch: false, cardinality: 'SINGLE', name: 'IntegerField' },
    },
  },
  tags: [],
  title: 'Resize',
  type: 'resize',
  useCache: true,
  version: '1.0.0',
};

// A holder rather than the template itself: a test swaps in a new object, which memoized tabs notice.
const templatesState = vi.hoisted(() => ({ templates: {} as Record<string, unknown> }));

templatesState.templates = { resize: template };

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useInvocationTemplatesSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ error: null, status: 'loaded', templates: templatesState.templates }),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        'widgets.workflow.inspectorTabs.outputs': 'Outputs',
        'widgets.workflow.latestOutput': 'Latest output',
        'widgets.workflow.noRunRecorded': 'This node has not run in this session.',
        'widgets.workflow.runStatus.completed': 'Completed',
        'widgets.workflow.runStatus.failed': 'Failed',
        'widgets.workflow.status': 'Status',
      })[key] ?? key,
  }),
}));

const node: WorkflowInvocationNode = {
  data: {
    inputs: {},
    isIntermediate: true,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type: 'resize',
    useCache: true,
    version: '1.0.0',
  },
  id: 'node-1',
  position: { x: 0, y: 0 },
  type: 'invocation',
};
const projectGraph: ProjectGraphState = { ...createProjectGraph('inspector-test'), nodes: [node] };
const projectSnapshot = {
  galleryValues: {},
  id: 'project-1',
  isWorkflowRunning: false,
  projectGraph,
  workflowValues: { inspectorTab: 'outputs' },
};

/** A node-execution port the test can advance between renders. */
const createExecutionPort = () => {
  const listeners = new Set<() => void>();
  let state: WorkflowNodeExecutionState | null = null;

  return {
    port: {
      get: (nodeId: string) => (nodeId === node.id ? state : null),
      subscribe: (_nodeId: string, listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    },
    set(next: WorkflowNodeExecutionState | null) {
      state = next;
      listeners.forEach((listener) => listener());
    },
  };
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
    preferences: { getSnapshot: () => ({}), subscribe: () => () => {} },
    project: { getSnapshot: () => projectSnapshot, subscribe: () => () => {} },
    registerModalHotkeyLayer: vi.fn(() => vi.fn()),
    widgets: { open: vi.fn(), patchValues: vi.fn() },
  }) as unknown as WorkflowUiAdapter;

describe('NodeInspector outputs tab', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    workflowSelectionStore.patchSnapshot({ hoveredNodeId: null, selectedNodeIds: [node.id], selectionRequest: null });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('shows the latest run: status, each declared output value, the saved image, and the raw result', async () => {
    const execution = createExecutionPort();

    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={createAdapter(execution.port)}>
            <NodeInspector projectGraph={projectGraph} />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );

    expect(host.textContent).toContain('This node has not run in this session.');
    expect(host.textContent).toContain('IntegerField');

    await act(() =>
      execution.set({
        error: null,
        latestOutput: { height: 768, type: 'image_output', width: 1024 },
        outputImageUrl: 'data:image/png;base64,',
        progress: null,
        progressMessage: null,
        status: 'completed',
      })
    );

    expect(host.textContent).not.toContain('This node has not run in this session.');
    expect(host.textContent).toContain('Completed');
    expect(host.textContent).toContain('1024');
    expect(host.textContent).toContain('768');
    expect(host.querySelector('img')?.getAttribute('src')).toBe('data:image/png;base64,');
    expect(host.querySelector('[aria-label="Latest output"]')?.textContent).toContain('"type": "image_output"');

    // The store carries the previous run's result through a failure; the tab must not pass it off as this run's.
    await act(() =>
      execution.set({
        error: 'Out of memory',
        latestOutput: { height: 768, type: 'image_output', width: 1024 },
        outputImageUrl: 'data:image/png;base64,',
        progress: null,
        progressMessage: null,
        status: 'failed',
      })
    );

    expect(host.textContent).toContain('Failed — Out of memory');
    expect(host.textContent).not.toContain('1024');
    expect(host.querySelector('img')).toBeNull();
  });
});

// Module scope: an adapter built inside the describe reads as a per-render object to the JSX prop lint.
const detailsSnapshot = { ...projectSnapshot, workflowValues: { inspectorTab: 'details' } };
const baseAdapter = createAdapter(createExecutionPort().port);
const detailsAdapter = {
  ...baseAdapter,
  project: { ...baseAdapter.project, getSnapshot: () => detailsSnapshot },
} as WorkflowUiAdapter;

describe('NodeInspector details tab node updates', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    workflowSelectionStore.patchSnapshot({ hoveredNodeId: null, selectedNodeIds: [node.id], selectionRequest: null });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    templatesState.templates = { resize: template };
  });

  const renderDetails = () =>
    act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={detailsAdapter}>
            <NodeInspector projectGraph={projectGraph} />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );
  const updateButton = () =>
    Array.from(host.querySelectorAll('button')).find((button) => button.textContent === 'nodes.updateNodeTo');

  it('offers an in-place update for a newer same-major template and explains an incompatible one', async () => {
    await renderDetails();
    expect(updateButton()).toBeUndefined();

    const newer = { ...template, version: '1.2.0' };

    templatesState.templates = { resize: newer };
    await renderDetails();

    updateButton()!.focus();
    await act(() => updateButton()!.click());
    expect(document.activeElement?.getAttribute('role')).toBe('tab');
    expect(detailsAdapter.commands.editGraph).toHaveBeenCalledWith({
      nodeIds: [node.id],
      templates: { resize: newer },
      type: 'updateNodes',
    });

    templatesState.templates = { resize: { ...template, version: '2.0.0' } };
    await renderDetails();
    expect(updateButton()).toBeUndefined();
    expect(host.textContent).toContain('nodes.nodeVersionIncompatible');
  });
});
