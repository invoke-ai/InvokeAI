import type { InvocationTemplate, ProjectGraphState, WorkflowInvocationNode } from '@features/workflow/contracts';
import type { WorkflowNodeExecutionState } from '@features/workflow/ui/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';
import type { ProjectGraphAction } from '@features/workflow/utility';

/* eslint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop -- each render mounts a fresh flow on purpose */
import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { setNodePreviewCollapsed } from '@features/workflow/ui/workflowUiStore';
import { buildCurrentImageNode, createProjectGraph, projectGraphReducer } from '@features/workflow/utility';
import { system } from '@theme/system';
import { applyNodeChanges, ReactFlow, type NodeChange } from '@xyflow/react';
import { createInstance } from 'i18next';
import { act, createRef, startTransition, useCallback, useEffect, useState, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ConnectorFlowNode } from './ConnectorFlowNode';
import { CurrentImageFlowNode } from './CurrentImageFlowNode';
import { toFlowEdges, toFlowNodes } from './flowAdapters';
import { InvocationFlowNode, WorkflowImageExportProvider } from './InvocationFlowNode';
import { exportWorkflowAsPng, WORKFLOW_EXPORT_TIMEOUT_MS } from './workflowImageExport';
import { WorkflowImageExportView } from './WorkflowImageExportView';

import '@xyflow/react/dist/style.css';

const exportMocks = vi.hoisted(() => ({
  progressImage: null as null | { dataUrl: string; height: number; width: number },
  toBlob: vi.fn(),
  translate: ((key: string) => key) as (key: string, options?: Record<string, unknown>) => string,
}));
vi.mock('html-to-image', () => ({ toBlob: exportMocks.toBlob }));
vi.mock('@platform/browser/downloadBlob', () => ({ downloadBlob: vi.fn() }));
vi.mock('@features/queue/react', () => ({ useProgressImage: () => exportMocks.progressImage }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: exportMocks.translate,
  }),
}));

const i18n = createInstance();
await i18n.init({
  fallbackLng: 'en',
  fallbackNS: 'translation',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: {
    en: { translation: await fetch('/locales/en.json').then((response) => response.json()) },
    fr: { translation: await fetch('/locales/fr.json').then((response) => response.json()) },
  },
});
exportMocks.translate = (key, options) => i18n.t(key, options as never) as unknown as string;

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
const currentImageNode = buildCurrentImageNode({ x: 20, y: 20 });
const currentImageGraph: ProjectGraphState = { ...createProjectGraph('current-image-test'), nodes: [currentImageNode] };
const currentImageNodes = toFlowNodes(currentImageGraph, []);
const nodeTypes = { connector: ConnectorFlowNode, current_image: CurrentImageFlowNode, invocation: InvocationFlowNode };

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
const PROJECT_ID = 'project-1';
const projectSnapshotFor = (graph: ProjectGraphState, galleryValues: Record<string, unknown> = {}) => ({
  activeWorkflow: { document: graph },
  activeWorkflowId: graph.id,
  galleryValues,
  id: PROJECT_ID,
  isWorkflowRunning: false,
  projectGraph: graph,
  workflowValues: {},
  workflows: [{ document: graph }],
});
const projectSnapshot = projectSnapshotFor(projectGraph);

/** Execution ports report this editor's own workflow as the run's origin, so node state is shown. */
const withOrigin = (
  port: Pick<WorkflowUiAdapter['nodeExecution'], 'get' | 'subscribe'>
): WorkflowUiAdapter['nodeExecution'] => ({
  ...port,
  getOrigin: () => ({ projectId: PROJECT_ID, workflowId: projectGraph.id }),
  subscribeOrigin: () => () => {},
});

const createAdapter = (
  nodeExecution: Pick<WorkflowUiAdapter['nodeExecution'], 'get' | 'subscribe'>,
  snapshot = projectSnapshot
): WorkflowUiAdapter =>
  ({
    capabilities: { getSnapshot: () => ({ canUseCache: true }), subscribe: () => () => {} },
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
    getProjectGraph: () => snapshot.projectGraph,
    nodeExecution: withOrigin(nodeExecution),
    notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
    openAddModels: vi.fn(),
    performance: {
      mark: vi.fn(),
      measure: vi.fn(),
      time: <T,>(_name: string, _source: unknown, callback: () => T) => callback(),
    },
    preferences: { getSnapshot: () => preferencesSnapshot, subscribe: () => () => {} },
    project: { getSnapshot: () => snapshot, subscribe: () => () => {} },
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
    exportMocks.progressImage = null;
    await i18n.changeLanguage('en');
  });

  const render = (adapter: WorkflowUiAdapter, zoom = 1, isExporting = false, nodes = flowNodes) =>
    act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowImageExportProvider isExporting={isExporting}>
            <WorkflowUiProvider adapter={adapter}>
              <ReactFlow
                defaultViewport={{ x: 0, y: 0, zoom }}
                edges={[]}
                minZoom={0.1}
                nodes={nodes}
                nodeTypes={nodeTypes}
              />
            </WorkflowUiProvider>
          </WorkflowImageExportProvider>
        </ChakraProvider>
      )
    );
  const image = () => host.querySelector<HTMLImageElement>('.react-flow__node img');
  const disclosure = () =>
    [...host.querySelectorAll<HTMLButtonElement>('.react-flow__node button[aria-expanded]')].find(
      (button) => button.textContent === 'Latest output'
    )!;
  const nodeHeight = () => host.querySelector<HTMLElement>('.react-flow__node')!.getBoundingClientRect().height;

  it('keeps node and field descriptions plus full output values in the exported graph clone', async () => {
    const longOutput = 'x'.repeat(60);
    const longOutputTitle = 'An output title that is deliberately long enough to wrap instead of truncating';
    const exportTemplate: InvocationTemplate = {
      ...template,
      category: 'Export category',
      classification: 'stable',
      description: 'Node export description',
      inputs: { a: { ...template.inputs.a!, description: 'Template field description' } },
      outputs: {
        value: {
          ...template.outputs.value!,
          description: 'Output export description',
          title: longOutputTitle,
        },
      },
    };
    const exportNode: WorkflowInvocationNode = {
      ...documentNode,
      data: {
        ...documentNode.data,
        inputs: {
          a: { description: 'Field export description', descriptionOverride: true, label: 'A', name: 'a', value: 1 },
        },
        isOpen: false,
        notes: 'Node export note',
      },
    };
    const exportConnector = {
      data: { label: '' },
      id: 'export-connector',
      position: { x: 380, y: 300 },
      type: 'connector' as const,
    };
    const exportGraph = { ...projectGraph, nodes: [exportNode, exportConnector] };
    const exportNodes = toFlowNodes(exportGraph, [], { preview: exportTemplate });
    const execution = createExecutionPort();
    const adapter = createAdapter(execution.port);
    let capturedClone: HTMLElement | undefined;
    exportMocks.toBlob.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, exportNodes);
    await act(() =>
      execution.set({
        error: null,
        latestOutput: { value: longOutput },
        outputImageUrl: null,
        progress: null,
        progressMessage: null,
        status: 'completed',
      })
    );
    await page.screenshot({ path: '../../../../../artifacts/workflow-export-content.png' });

    const flowElement = host.querySelector<HTMLElement>('.react-flow')!;
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement,
      workflowName: 'Export content',
    });

    expect(capturedClone?.textContent).toContain('Node export description');
    expect(capturedClone?.textContent).toContain('Node export note');
    expect(capturedClone?.textContent).toContain('Field export description');
    expect(capturedClone?.textContent).toContain('Output export description');
    expect(capturedClone?.textContent).toContain('Type: preview');
    expect(capturedClone?.textContent).toContain('Node pack: invokeai');
    expect(capturedClone?.textContent).toContain('Version: 1.0.0');
    expect(capturedClone?.textContent).toContain('Classification: stable');
    expect(capturedClone?.textContent).toContain('Category: Export category');
    expect(capturedClone?.textContent).toContain('Field: a');
    expect(capturedClone?.textContent).toContain('Type: Integer');
    expect(capturedClone?.textContent).toContain('Connection only');
    expect(capturedClone?.textContent).toContain('Output');
    expect(capturedClone?.textContent).toContain('Any input');
    expect(capturedClone?.textContent).toContain('Connector: Any input → Any output');
    expect(capturedClone?.textContent).toContain('Any output');
    expect(capturedClone?.textContent).toContain(longOutput);
    const outputTitle = capturedClone?.querySelector<HTMLElement>('[data-workflow-export-output-title="true"]');
    expect(outputTitle?.textContent).toBe(longOutputTitle);
    expect(outputTitle && getComputedStyle(outputTitle).whiteSpace).not.toBe('nowrap');
    expect(capturedClone?.querySelectorAll('[data-workflow-export-connector-tooltip="true"]')).toHaveLength(5);
    expect(capturedClone?.querySelectorAll('[data-workflow-export-content="true"]').length).toBeGreaterThanOrEqual(4);
  });

  it('keeps running progress in the editor but omits it from the exported graph clone', async () => {
    const execution = createExecutionPort();
    const adapter = createAdapter(execution.port);
    let capturedClone: HTMLElement | undefined;
    exportMocks.toBlob.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter);
    await act(() =>
      execution.set({
        error: null,
        latestOutput: null,
        outputImageUrl: null,
        progress: 0.5,
        progressMessage: null,
        status: 'running',
      })
    );
    expect(host.querySelector('[data-node-progress-strip="true"]')).not.toBeNull();

    await render(adapter, 1, true);

    const flowElement = host.querySelector<HTMLElement>('.react-flow')!;
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement,
      workflowName: 'Running workflow',
    });

    expect(capturedClone?.querySelector('[data-node-progress-strip="true"]')).toBeNull();
  });

  it('uses the saved current image in exports while the editor shows the live generation frame', async () => {
    const liveImageUrl = 'data:image/png;base64,bGl2ZQ==';
    const savedImageUrl = 'data:image/png;base64,c2F2ZWQ=';
    let capturedClone: HTMLElement | undefined;
    exportMocks.progressImage = { dataUrl: liveImageUrl, height: 2, width: 2 };
    exportMocks.toBlob.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });
    const savedImage = {
      height: 2,
      imageName: 'saved.png',
      imageUrl: savedImageUrl,
      queuedAt: '2026-01-01T00:00:00Z',
      sourceQueueItemId: 'queue-item',
      thumbnailUrl: savedImageUrl,
      width: 2,
    };
    const adapter = createAdapter(
      createExecutionPort().port,
      projectSnapshotFor(currentImageGraph, { recentImages: [savedImage] })
    );

    await render(adapter, 1, false, currentImageNodes);
    expect(host.querySelector<HTMLImageElement>('.react-flow__node img')?.src).toBe(liveImageUrl);

    await render(adapter, 1, true, currentImageNodes);
    const flowElement = host.querySelector<HTMLElement>('.react-flow')!;
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement,
      workflowName: 'Current image',
    });

    const exportedImage = capturedClone?.querySelector<HTMLImageElement>('.react-flow__node img');
    expect(exportedImage?.src).toBe(savedImageUrl);
    expect(exportedImage?.src).not.toBe(liveImageUrl);
  });

  it('localizes exported field and connector metadata using the selected language catalog', async () => {
    await i18n.changeLanguage('fr');
    const batchType = { ...template.inputs.a!.type, batch: true };
    const localizedTemplate: InvocationTemplate = {
      ...template,
      inputs: { a: { ...template.inputs.a!, required: true, type: batchType } },
      outputs: { value: { ...template.outputs.value!, type: batchType } },
    };
    const localizedNode: WorkflowInvocationNode = {
      ...documentNode,
      data: { ...documentNode.data, inputs: { a: { label: 'A', name: 'a', value: null } } },
    };
    const connector = {
      data: { label: '' },
      id: 'localized-connector',
      position: { x: 380, y: 300 },
      type: 'connector' as const,
    };
    const batchConnector = {
      data: { label: '' },
      id: 'localized-batch-connector',
      position: { x: 500, y: 300 },
      type: 'connector' as const,
    };
    const graph = { ...projectGraph, nodes: [localizedNode, connector, batchConnector] };
    const nodes = toFlowNodes(graph, [], { preview: localizedTemplate });
    const localizedConnector = nodes.find(
      (node) => node.type === 'connector' && node.data.documentNode.id === 'localized-batch-connector'
    );
    if (localizedConnector?.type === 'connector') {
      localizedConnector.data.inputFieldType = batchType;
      localizedConnector.data.outputFieldType = batchType;
    }
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
    let capturedClone: HTMLElement | undefined;
    exportMocks.toBlob.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, nodes);
    const flowElement = host.querySelector<HTMLElement>('.react-flow')!;
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement,
      workflowName: 'Workflow localisé',
    });

    const exportedText = capturedClone?.textContent ?? '';
    expect(exportedText).toContain('Champ : a');
    expect(exportedText).toContain('Type : Integer');
    expect(exportedText).toContain('Obligatoire');
    expect(exportedText).toContain('Connexion uniquement');
    expect(exportedText).toContain('Integer en lot');
    expect(exportedText).not.toContain(' batch');
    expect(exportedText).toContain('N’importe quelle entrée');
    expect(exportedText).toContain('Connecteur : N’importe quelle entrée → N’importe quelle sortie');
    expect(exportedText).toContain('N’importe quelle sortie');
  });

  it('marks a failed node in the export without including its backend diagnostic', async () => {
    const execution = createExecutionPort(callNodeId);
    execution.set({
      error: 'Child node failed',
      latestOutput: null,
      outputImageUrl: null,
      progress: null,
      progressMessage: null,
      status: 'failed',
    });
    const adapter = createAdapter(execution.port);
    let capturedClone: HTMLElement | undefined;
    exportMocks.toBlob.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, callFlowNodes);
    await page.screenshot({ path: '../../../../../artifacts/workflow-export-failed-label.png' });
    const flowElement = host.querySelector<HTMLElement>('.react-flow')!;
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement,
      workflowName: 'Failed workflow',
    });

    expect(capturedClone?.querySelector('[data-node-status-indicator="true"]')?.getAttribute('style')).toContain(
      'display: none'
    );
    expect(capturedClone?.textContent).toContain('Failed');
    expect(capturedClone?.textContent).not.toContain('Child node failed');
    expect(capturedClone?.textContent).not.toContain('Child workflow error');
  });

  it('keeps the visible collapsed node unchanged while an expanded offscreen snapshot is rasterizing', async () => {
    const node = { ...documentNode, data: { ...documentNode.data, isOpen: false, notes: 'Snapshot metadata' } };
    const directNode = { ...documentNode, id: 'direct-node', position: { x: 20, y: 100 } };
    const graph = { ...projectGraph, nodes: [node, directNode] };
    const directTemplate = { ...template, inputs: { a: { ...template.inputs.a!, input: 'direct' as const } } };
    const nodes = toFlowNodes(graph, [], { preview: directTemplate });
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
    const exportRef = createRef<HTMLDivElement>();
    host.style.position = 'relative';
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <div data-visible-editor style={{ height: '100%', width: '100%' }}>
              <ReactFlow nodes={nodes} edges={[]} nodeTypes={nodeTypes} />
            </div>
            <WorkflowImageExportView containerRef={exportRef} nodes={nodes} edges={[]} nodeTypes={nodeTypes} />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );
    const visible = host.querySelector<HTMLElement>('[data-visible-editor]')!;
    const visibleNode = visible.querySelector<HTMLElement>('.react-flow__node')!;
    const beforeHeight = visibleNode.getBoundingClientRect().height;
    expect(beforeHeight).toBeGreaterThan(0);
    let finish: (blob: Blob) => void = () => {};
    let capturedClone: HTMLElement | undefined;
    exportMocks.toBlob.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return new Promise<Blob>((resolve) => {
        finish = resolve;
      });
    });
    const capture = exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: exportRef.current!.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Isolated snapshot',
    });
    await vi.waitFor(() => expect(capturedClone).toBeDefined());
    expect(visible.textContent).not.toContain('Snapshot metadata');
    expect(visible.querySelector('button[aria-label="Expand node"]')).not.toBeNull();
    expect(visibleNode.getBoundingClientRect().height).toBe(beforeHeight);
    const visibleInput = visible.querySelector<HTMLInputElement>('input');
    const offscreenInput = exportRef.current!.querySelector<HTMLInputElement>('input');
    expect(visibleInput).not.toBeNull();
    expect(offscreenInput).not.toBeNull();
    expect(visibleInput?.id).not.toBe(offscreenInput?.id);
    expect(capturedClone?.textContent).toContain('Snapshot metadata');
    await page.screenshot({ path: '../../../../../artifacts/workflow-export-pending.png' });
    finish(new Blob(['png'], { type: 'image/png' }));
    await capture;
  });

  it.each(['running', 'failed'] as const)('omits retained results from a %s retry snapshot', async (status) => {
    const execution = createExecutionPort();
    const completed = {
      error: null,
      latestOutput: { value: 'previous result' },
      outputImageUrl: 'data:image/png;base64,cHJldmlvdXM=',
      progress: null,
      progressMessage: null,
      status: 'completed' as const,
    };
    execution.set(completed);
    const adapter = createAdapter(execution.port);
    await render(adapter, 1, true);
    expect(host.textContent).toContain('previous result');
    await act(() => execution.set({ ...completed, status }));
    let capturedClone: HTMLElement | undefined;
    exportMocks.toBlob.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Retry',
    });
    expect(capturedClone?.textContent).not.toContain('previous result');
    expect(capturedClone?.querySelector('img')).toBeNull();
    if (status === 'failed') {
      expect(capturedClone?.textContent).toContain('Failed');
    }
  });

  it('blocks a third capture after two timeouts and recovers when one rasterization settles', async () => {
    vi.useFakeTimers();
    exportMocks.toBlob.mockClear();
    const finishRasterizations: Array<(blob: Blob | null) => void> = [];
    let signalBothStarted: () => void = () => {};
    const bothStarted = new Promise<void>((resolve) => {
      signalBothStarted = resolve;
    });
    exportMocks.toBlob.mockImplementation(
      () =>
        new Promise<Blob | null>((resolve) => {
          finishRasterizations.push(resolve);
          if (finishRasterizations.length === 2) {
            signalBothStarted();
          }
        })
    );
    const execution = createExecutionPort();
    const adapter = createAdapter(execution.port);
    const exportOptions = {
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: undefined as unknown as HTMLElement,
      workflowName: 'Stalled workflow',
    };

    try {
      await render(adapter, 1, true);
      exportOptions.flowElement = host.querySelector<HTMLElement>('.react-flow')!;
      const firstExport = exportWorkflowAsPng(exportOptions);
      const secondExport = exportWorkflowAsPng(exportOptions);
      const firstTimedOut = expect(firstExport).rejects.toThrow('timed out');
      const secondTimedOut = expect(secondExport).rejects.toThrow('timed out');

      await bothStarted;
      expect(exportMocks.toBlob).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(WORKFLOW_EXPORT_TIMEOUT_MS);
      await Promise.all([firstTimedOut, secondTimedOut]);

      await expect(exportWorkflowAsPng(exportOptions)).rejects.toThrow('still running');
      expect(exportMocks.toBlob).toHaveBeenCalledTimes(2);

      finishRasterizations[0]!(null);
      await vi.advanceTimersByTimeAsync(0);
      exportMocks.toBlob.mockResolvedValueOnce(new Blob(['png'], { type: 'image/png' }));
      await exportWorkflowAsPng(exportOptions);
      expect(exportMocks.toBlob).toHaveBeenCalledTimes(3);

      finishRasterizations[1]!(null);
      await vi.advanceTimersByTimeAsync(0);
    } finally {
      finishRasterizations.forEach((finish) => finish(null));
      await vi.advanceTimersByTimeAsync(0);
      vi.useRealTimers();
    }
  });

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

  it('renders full node content for image export when the viewport is zoomed out', async () => {
    const execution = createExecutionPort();
    const adapter = createAdapter(execution.port);
    execution.set(completed(outputImage(400, 100)));

    await render(adapter, 0.3, true);

    await vi.waitFor(() => expect(image()).not.toBeNull());
  });

  it('renders full fields for compact nodes during image export', async () => {
    const execution = createExecutionPort();
    const adapter = createAdapter(execution.port);
    const compactNode = {
      ...documentNode,
      data: { ...documentNode.data, inputs: { a: { label: '', name: 'a', value: 42 } } },
    };
    const compactTemplates = {
      preview: { ...template, inputs: { a: { ...template.inputs.a!, input: 'any' as const } } },
    };
    const compactNodes = toFlowNodes({ ...projectGraph, nodes: [compactNode] }, [], compactTemplates, undefined, true);

    await render(adapter, 1, true, compactNodes);

    expect(host.querySelector('[data-node-input-field-title="true"]')?.textContent).toContain('A');
    expect(host.querySelector<HTMLInputElement>('.react-flow__node input')?.value).toBe('42');
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
    const base = createAdapter({ get: () => null, subscribe: () => () => {} });
    const adapter = {
      ...base,
      commands: { ...base.commands, editGraph: store.editGraph, undo: store.undo },
      getProjectGraph: store.getSnapshot,
      project: {
        getSnapshot: () => projectSnapshotFor(store.getSnapshot()),
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
  const updateIcon = () => host.querySelector('.react-flow__node [role="img"][aria-label*="Version"]');
  const borderColor = () => getComputedStyle(host.querySelector<HTMLElement>('.react-flow__node > div')!).borderColor;

  it('marks a node whose template moved on with a warning border and an update tooltip, and leaves a current one alone', async () => {
    await render(flowNodes);
    expect(updateIcon()).toBeNull();
    const currentBorder = borderColor();

    await render(toFlowNodes(projectGraph, [], { preview: { ...template, version: '1.1.0' } }));
    expect(updateIcon()?.getAttribute('aria-label')).toBe('Version 1.0.0 → 1.1.0: update available');
    expect(borderColor()).not.toBe(currentBorder);

    await render(toFlowNodes(projectGraph, [], { preview: { ...template, version: '2.0.0' } }));
    expect(updateIcon()?.getAttribute('aria-label')).toBe(
      'Version 1.0.0 cannot be updated to 2.0.0; delete and re-add the node'
    );
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
    expect(header().textContent).toContain('(no group)');
  });
});
