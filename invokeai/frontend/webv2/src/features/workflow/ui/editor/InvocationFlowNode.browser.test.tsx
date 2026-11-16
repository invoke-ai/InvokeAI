import type {
  InvocationTemplate,
  ProjectGraphState,
  WorkflowEdge,
  WorkflowInvocationNode,
} from '@features/workflow/contracts';
import type { WorkflowNodeExecutionState } from '@features/workflow/ui/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';
import type { ProjectGraphAction } from '@features/workflow/utility';

import openapi from '@api/openapi.json';
/* eslint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop -- each render mounts a fresh flow on purpose */
import { Box, ChakraProvider } from '@chakra-ui/react';
import { DndContext, type DragStartEvent } from '@dnd-kit/core';
import { galleryImageUrls } from '@features/gallery/utility';
import {
  savedWorkflowDetailQueryKey,
  savedWorkflowPickerQueryOptions,
} from '@features/workflow/data/savedWorkflowQueries';
import { parseOpenApiToTemplates } from '@features/workflow/data/templates';
import { flowThemeCss } from '@features/workflow/ui/graph-preview/flowTheme';
import { WorkflowImageExportProvider } from '@features/workflow/ui/nodeChrome';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import {
  buildInvocationNode,
  buildCurrentImageNode,
  buildNotesNode,
  createProjectGraph,
  projectGraphReducer,
} from '@features/workflow/utility';
import { downloadBlob } from '@platform/browser/downloadBlob';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { applyNodeChanges, ReactFlow, type NodeChange } from '@xyflow/react';
import { createInstance } from 'i18next';
import { act, createRef, startTransition, useCallback, useEffect, useState, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { commands, userEvent } from 'vitest/browser';

import type { rasterizeWorkflowImage as RasterizeWorkflowImage } from './workflowImageRaster';

import { ConnectorFlowNode } from './ConnectorFlowNode';
import { CurrentImageFlowNode } from './CurrentImageFlowNode';
import { toFlowEdges, toFlowNodes } from './flowAdapters';
import { InvocationFlowNode } from './InvocationFlowNode';
import { LoopBodyBoundaryOverlay } from './LoopBodyBoundaryOverlay';
import { NotesFlowNode } from './NotesFlowNode';
import { WorkflowEdge as WorkflowEdgeComponent } from './WorkflowEdge';
import {
  EXPORT_SCALE,
  exportWorkflowAsPng,
  getWorkflowContentBounds,
  WORKFLOW_EXPORT_TIMEOUT_MS,
} from './workflowImageExport';
import { WorkflowImageExportView } from './WorkflowImageExportView';

import '@xyflow/react/dist/style.css';

declare module 'vitest/browser' {
  interface BrowserCommands {
    mouseDragPath: (points: readonly { x: number; y: number }[], frameWidth: number) => Promise<void>;
  }
}

const exportMocks = vi.hoisted(() => ({
  progressImage: null as null | { dataUrl: string; height: number; width: number },
  rasterize: vi.fn(),
  translate: ((key: string) => key) as (key: string, options?: Record<string, unknown>) => string,
}));
vi.mock('./workflowImageRaster', () => ({ rasterizeWorkflowImage: exportMocks.rasterize }));
vi.mock('@platform/browser/downloadBlob', () => ({ downloadBlob: vi.fn() }));
vi.mock('@features/queue/react', () => ({ useProgressImage: () => exportMocks.progressImage }));
vi.mock('@features/generation/queries', () => ({
  promptTemplatesQueryOptions: () => ({
    queryFn: () => Promise.resolve([]),
    queryKey: ['generation', 'promptTemplates', 'list'],
    staleTime: 30_000,
  }),
}));

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
const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

const NODE_ID = 'preview-node';

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
const nodeTypes = {
  connector: ConnectorFlowNode,
  current_image: CurrentImageFlowNode,
  invocation: InvocationFlowNode,
  notes: NotesFlowNode,
};

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

const completed = (outputImageName: string): WorkflowNodeExecutionState => ({
  error: null,
  latestOutput: null,
  outputImageName,
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
    widgets: { open: vi.fn(), patchValues: vi.fn() },
  }) as unknown as WorkflowUiAdapter;

describe('InvocationFlowNode chrome and export', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    queryClient.clear();
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
          <QueryClientProvider client={queryClient}>
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
          </QueryClientProvider>
        </ChakraProvider>
      )
    );
  const image = () => host.querySelector<HTMLImageElement>('.react-flow__node img');
  const nodeHeight = () => host.querySelector<HTMLElement>('.react-flow__node')!.getBoundingClientRect().height;

  it('disables both HiDiffusion numeric controls under one Auto switch and retains manual values', async () => {
    const denoiseTemplates = parseOpenApiToTemplates(openapi);
    const node = buildInvocationNode(denoiseTemplates.denoise_latents!, { x: 0, y: 0 });
    node.data.inputs.hidiffusion!.value = true;
    node.data.inputs.hidiffusion_t1_ratio!.value = 0.7;
    node.data.inputs.hidiffusion_t2_ratio!.value = 0.2;
    for (const auto of [true, false]) {
      node.data.inputs.hidiffusion_auto_ratios!.value = auto;
      const graph = { ...createProjectGraph('hidiffusion'), nodes: [structuredClone(node)] };
      const adapter = createAdapter(createExecutionPort(node.id).port, projectSnapshotFor(graph));
      await render(adapter, 1, false, toFlowNodes(graph, [], denoiseTemplates));
      for (const [name, value] of [
        ['hidiffusion_t1_ratio', '0.7'],
        ['hidiffusion_t2_ratio', '0.2'],
      ]) {
        const control = host.querySelector<HTMLInputElement>(`input[id="${node.id}-${name}-value-number-input"]`);
        expect(control).not.toBeNull();
        expect(control!.disabled).toBe(auto);
        expect(control!.value).toBe(value);
      }
      expect(
        host.querySelector<HTMLInputElement>(`input[id="${node.id}-hidiffusion_t1_ratio-value-number-input"]`)!.min
      ).toBe(auto ? '0' : '0.2');
      expect(
        host.querySelector<HTMLInputElement>(`input[id="${node.id}-hidiffusion_t2_ratio-value-number-input"]`)!.max
      ).toBe(auto ? '1' : '0.7');
    }
  });

  it('keeps field-description editing available in the workflow editor', async () => {
    const adapter = createAdapter(createExecutionPort().port);
    await render(adapter);

    const trigger = host.querySelector<HTMLButtonElement>('button[aria-label="Edit field description"]');
    expect(trigger).not.toBeNull();
    await userEvent.click(trigger!);
    const description = await vi.waitFor(() => {
      const textarea = document.querySelector<HTMLTextAreaElement>('textarea[aria-label="Field description"]');
      expect(textarea).not.toBeNull();
      return textarea!;
    });
    await userEvent.fill(description, 'Authored field guidance');

    expect(adapter.commands.editGraph).toHaveBeenCalledWith({
      description: 'Authored field guidance',
      fieldName: 'a',
      nodeId: NODE_ID,
      type: 'setFieldDescription',
    });
  });

  it('sizes a snapshot by the nodes it renders, not by the live editor’s bounds', async () => {
    // Live, the node is collapsed to its header; the snapshot draws every node open, so it is taller.
    const collapsedNodes = toFlowNodes(
      { ...projectGraph, nodes: [{ ...documentNode, data: { ...documentNode.data, isOpen: false } }] },
      [],
      templates
    );
    const adapter = createAdapter(createExecutionPort().port);
    let capturedHeight: number | undefined;
    exportMocks.rasterize.mockImplementation((_clone: HTMLElement, options: { height: number }) => {
      capturedHeight = options.height;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, false, collapsedNodes);
    const liveRect = host.querySelector<HTMLElement>('.react-flow__node')!.getBoundingClientRect();
    const liveBounds = { x: 20, y: 20, width: liveRect.width, height: liveRect.height };
    await render(adapter, 1, true, collapsedNodes);
    const snapshotRect = host.querySelector<HTMLElement>('.react-flow__node')!.getBoundingClientRect();
    expect(snapshotRect.height).toBeGreaterThan(liveRect.height);

    await exportWorkflowAsPng({
      bounds: liveBounds,
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Collapsed live',
    });

    expect(capturedHeight).toBe(Math.ceil(snapshotRect.height + 200) * EXPORT_SCALE);
  });

  it.each([
    { kind: 'plain', uiComponent: null, value: 'Authored text' },
    { kind: 'multiline', uiComponent: 'textarea', value: 'First line\nSecond line' },
    { kind: 'long', uiComponent: 'textarea', value: 'UnbrokenText'.repeat(50) },
    { kind: 'empty', uiComponent: null, value: '' },
  ] as const)('preserves static text field entry boxes for $kind fields', async ({ uiComponent, value }) => {
    const textTemplate: InvocationTemplate = {
      ...template,
      inputs: {
        a: {
          ...template.inputs.a!,
          input: 'direct',
          type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
          uiComponent,
        },
      },
    };
    const textNode: WorkflowInvocationNode = {
      ...documentNode,
      data: { ...documentNode.data, inputs: { a: { label: '', name: 'a', value } } },
    };
    const graph = { ...projectGraph, nodes: [textNode] };
    const nodes = toFlowNodes(graph, [], { preview: textTemplate });
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
    let capturedText: string | null | undefined;
    let capturedBorderWidth: string | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      const entryBox = clone.querySelector<HTMLElement>('[data-workflow-export-field-value="true"]');
      capturedText = entryBox?.textContent;
      capturedBorderWidth = entryBox ? getComputedStyle(entryBox).borderTopWidth : undefined;
      expect(entryBox).not.toBeNull();
      expect(entryBox!.scrollWidth).toBeLessThanOrEqual(entryBox!.clientWidth);
      expect(clone.querySelector('input, textarea, button')).toBeNull();
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, nodes);
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Text entry boxes',
    });

    expect(capturedText).toBe(value);
    expect(capturedBorderWidth).toBe('1px');
    expect(adapter.commands.editGraph).not.toHaveBeenCalled();
  });

  it('renders authored notes as static text in snapshot exports', async () => {
    const noteBody = Array.from({ length: 60 }, (_, index) => `Snapshot note line ${index + 1}`).join('\n');
    const authoredNote = {
      ...buildNotesNode({ x: 20, y: 20 }),
      data: { label: 'Planning note', notes: noteBody },
    };
    const emptyNote = {
      ...buildNotesNode({ x: 320, y: 20 }),
      data: { label: 'Empty note', notes: '' },
    };
    const graph = { ...projectGraph, nodes: [authoredNote, emptyNote] };
    const nodes = toFlowNodes(graph, []);
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
    let capturedClone: HTMLElement | undefined;
    let capturedOptions: { height?: number } | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement, options: { height?: number }) => {
      capturedClone = clone;
      capturedOptions = options;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, nodes);
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 500, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Notes',
    });

    expect(capturedClone?.textContent).toContain('Planning note');
    expect(capturedClone?.textContent).toContain('Snapshot note line 60');
    expect(capturedClone?.textContent).toContain('Empty note');
    expect(capturedClone?.textContent).not.toContain('Write a note');
    expect(capturedClone?.querySelector('input[aria-label="Note title"]')).toBeNull();
    expect(capturedClone?.querySelector('textarea[aria-label="Note text"]')).toBeNull();
    expect(capturedOptions?.height).toBeGreaterThan(500);
  });

  it('omits editor checkboxes from executable image-output nodes in snapshot exports', async () => {
    const imageTemplate: InvocationTemplate = {
      ...template,
      inputs: {},
      outputs: {
        image: {
          description: '',
          name: 'image',
          title: 'Image',
          type: { batch: false, cardinality: 'SINGLE', name: 'ImageField' },
        },
      },
      type: 'txt2img',
    };
    const imageNode = { ...documentNode, data: { ...documentNode.data, type: 'txt2img' } };
    const graph = { ...projectGraph, nodes: [imageNode] };
    const nodes = toFlowNodes(graph, [], { txt2img: imageTemplate });
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, nodes);
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Image output',
    });

    expect(capturedClone?.textContent).not.toContain(i18n.t('nodes.useCache'));
    expect(capturedClone?.textContent).not.toContain(i18n.t('nodes.saveToGallery'));
    expect(capturedClone?.querySelector('input[type="checkbox"]')).toBeNull();
  });

  it('exports authored field labels without tooltip metadata or execution results', async () => {
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
        version: '0.9.0',
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
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, exportNodes);
    await act(() =>
      execution.set({
        error: null,
        latestOutput: { value: longOutput },
        outputImageName: 'data:image/png;base64,cHJldmlldw==',
        progress: null,
        progressMessage: null,
        status: 'completed',
      })
    );

    const flowElement = host.querySelector<HTMLElement>('.react-flow')!;
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement,
      workflowName: 'Export content',
    });

    expect(capturedClone?.textContent).not.toContain('Node export description');
    expect(capturedClone?.textContent).not.toContain('Node export note');
    expect(capturedClone?.textContent).not.toContain('Field export description');
    expect(capturedClone?.textContent).not.toContain('Output export description');
    expect(capturedClone?.textContent).not.toContain('Type: preview');
    expect(capturedClone?.textContent).not.toContain('Node pack: invokeai');
    expect(capturedClone?.textContent).not.toContain('Version: 1.0.0');
    expect(capturedClone?.textContent).not.toContain('Field: a');
    expect(capturedClone?.textContent).not.toContain('Type: Integer');
    expect(capturedClone?.textContent).not.toContain('Connection only');
    expect(capturedClone?.textContent).not.toContain('Any input');
    expect(capturedClone?.textContent).not.toContain('Connector: Any input → Any output');
    expect(capturedClone?.textContent).not.toContain('Any output');
    expect(capturedClone?.textContent).not.toContain(longOutput);
    expect(capturedClone?.querySelector('.react-flow__node img')).toBeNull();
    expect(capturedClone?.querySelector('[data-node-input-field-title="true"]')?.textContent).toContain('A');
    const outputTitle = capturedClone?.querySelector<HTMLElement>('[data-workflow-export-output-title="true"]');
    expect(outputTitle?.textContent).toBe(longOutputTitle);
    expect(outputTitle && getComputedStyle(outputTitle).whiteSpace).not.toBe('nowrap');
    expect(capturedClone?.querySelectorAll('[data-workflow-export-connector-tooltip="true"]')).toHaveLength(0);
    expect(capturedClone?.querySelectorAll('[data-workflow-export-content="true"]')).toHaveLength(0);
    expect(capturedClone?.querySelector('[data-node-info-icon="true"]')).toBeNull();
    expect(capturedClone?.querySelector('svg[aria-label]')).toBeNull();
    expect(capturedClone?.querySelector('button[aria-label="Collapse node"]')).toBeNull();
  });

  it('keeps long authored node titles on one line in the static export', async () => {
    const invocationLabel = 'Authored invocation label '.repeat(24);
    const currentImageLabel = `Authored current image ${'unbroken-label-'.repeat(24)}`;
    const invocationNode: WorkflowInvocationNode = {
      ...documentNode,
      data: { ...documentNode.data, label: invocationLabel },
      id: 'long-label-node',
    };
    const imageNode = {
      ...buildCurrentImageNode({ x: 420, y: 20 }),
      data: { label: currentImageLabel },
      id: 'long-current-image-node',
    };
    const unknownNode: WorkflowInvocationNode = {
      ...invocationNode,
      data: { ...invocationNode.data, type: 'unknown-node-type' },
      id: 'long-unknown-node',
      position: { x: 20, y: 200 },
    };
    const notesNode = {
      ...buildNotesNode({ x: 420, y: 200 }),
      data: { label: invocationLabel, notes: 'Authored note body' },
      id: 'long-notes-node',
    };
    const graph = { ...projectGraph, nodes: [invocationNode, imageNode, unknownNode, notesNode] };
    const nodes = toFlowNodes(graph, [], { preview: template });
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, nodes);
    const flowElement = host.querySelector<HTMLElement>('.react-flow')!;
    const invocationTitle = host.querySelector<HTMLElement>(
      '.react-flow__node[data-id="long-label-node"] [data-workflow-export-node-title="true"]'
    );
    const imageTitle = host.querySelector<HTMLElement>(
      '.react-flow__node[data-id="long-current-image-node"] [data-workflow-export-node-title="true"]'
    );
    expect(invocationTitle?.textContent).toBe(invocationLabel);
    expect(getComputedStyle(invocationTitle!).whiteSpace).toBe('nowrap');
    expect(imageTitle?.textContent).toBe(currentImageLabel);
    expect(getComputedStyle(imageTitle!).whiteSpace).toBe('nowrap');
    const titles = host.querySelectorAll<HTMLElement>('[data-workflow-export-node-title="true"]');
    expect(titles).toHaveLength(4);
    for (const title of titles) {
      expect(getComputedStyle(title).whiteSpace).toBe('nowrap');
      expect(title.textContent?.length).toBeGreaterThan(100);
      // One line: every fragment of its text shares a top. (Its box is trimmed to the letters, so its ink overflows.)
      const lineTops = new Set<number>();
      const walker = document.createTreeWalker(title, NodeFilter.SHOW_TEXT);
      for (let text = walker.nextNode(); text; text = walker.nextNode()) {
        const range = document.createRange();
        range.selectNodeContents(text);
        for (const rect of range.getClientRects()) {
          lineTops.add(Math.round(rect.top));
        }
      }
      expect(lineTops.size).toBe(1);
    }
    const truncatedHead = invocationTitle!.firstElementChild as HTMLElement;
    expect(getComputedStyle(truncatedHead).textOverflow).toBe('ellipsis');
    expect(truncatedHead.scrollWidth).toBeGreaterThan(truncatedHead.clientWidth);
    expect(invocationTitle?.dataset.workflowExportStaticNodeContent).toBe('true');
    expect(imageTitle?.dataset.workflowExportStaticNodeContent).toBe('true');

    const bounds = getWorkflowContentBounds(flowElement, { x: 20, y: 20, width: 720, height: 100 });
    expect(bounds.height).toBeGreaterThan(100);
    await exportWorkflowAsPng({
      bounds,
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement,
      workflowName: 'Long labels',
    });

    expect(capturedClone?.textContent).toContain(invocationLabel);
    expect(capturedClone?.textContent).toContain(currentImageLabel);
  });

  it.each(['detail', 'owned', 'shared', 'unavailable', 'refreshed'] as const)(
    'exports the selected saved workflow name from %s metadata without picker controls',
    async (source) => {
      const workflowId = 'selected-child';
      const name = 'Landscape child workflow';
      const initialName = source === 'refreshed' ? 'Previous child workflow' : name;
      if (source === 'refreshed') {
        queryClient.setQueryData(
          savedWorkflowDetailQueryKey(workflowId),
          { name: initialName, workflow_id: workflowId },
          { updatedAt: 100 }
        );
        queryClient.setQueryData(
          savedWorkflowPickerQueryOptions({ page: 0, query: 'Previous' }).queryKey,
          {
            pageParams: [0],
            pages: [{ items: [{ name: initialName, workflow_id: workflowId }], page: 0, pages: 1, total: 1 }],
          },
          { updatedAt: 200 }
        );
      } else if (source === 'detail') {
        queryClient.setQueryData(savedWorkflowDetailQueryKey(workflowId), { name, workflow_id: workflowId });
      } else if (source !== 'unavailable') {
        const params = { page: 0, query: 'Landscape', isPublic: source === 'shared' };
        queryClient.setQueryData(savedWorkflowPickerQueryOptions(params).queryKey, {
          pageParams: [0],
          pages: [{ items: [{ name, workflow_id: workflowId }], page: 0, pages: 1, total: 1 }],
        });
      }
      const namedCallTemplate: InvocationTemplate = {
        ...callTemplate,
        inputs: {
          workflow_id: {
            ...template.inputs.a!,
            input: 'direct',
            name: 'workflow_id',
            title: 'Workflow',
            type: { batch: false, cardinality: 'SINGLE', name: 'SavedWorkflowField' },
          },
        },
      };
      const namedCallNode: WorkflowInvocationNode = {
        ...callNode,
        data: {
          ...callNode.data,
          inputs: { workflow_id: { label: '', name: 'workflow_id', value: workflowId } },
        },
      };
      const graph = { ...callProjectGraph, nodes: [namedCallNode] };
      const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
      const fetchSpy = vi.spyOn(globalThis, 'fetch');
      try {
        await render(adapter, 1, true, toFlowNodes(graph, [], { call_saved_workflow: namedCallTemplate }));
        const titleHead = host.querySelector<HTMLElement>('[data-workflow-export-node-title="true"] > span');
        expect(titleHead).not.toBeNull();
        expect(titleHead!.scrollWidth).toBeLessThanOrEqual(titleHead!.clientWidth);
        expect(host.querySelector('[data-workflow-export-field-value="true"]')?.textContent).toBe(
          source === 'unavailable' ? workflowId : initialName
        );
        if (source === 'refreshed') {
          await act(() => {
            queryClient.setQueryData(
              savedWorkflowPickerQueryOptions({ page: 0, query: 'Landscape' }).queryKey,
              {
                pageParams: [0],
                pages: [{ items: [{ name, workflow_id: workflowId }], page: 0, pages: 1, total: 1 }],
              },
              { updatedAt: 300 }
            );
          });
          expect(host.querySelector('[data-workflow-export-field-value="true"]')?.textContent).toBe(initialName);
          await act(() => {
            queryClient.setQueryData(
              savedWorkflowDetailQueryKey(workflowId),
              {
                name: 'Renamed child workflow',
                workflow_id: workflowId,
              },
              { updatedAt: 400 }
            );
          });
          expect(host.querySelector('[data-workflow-export-field-value="true"]')?.textContent).toBe(
            'Renamed child workflow'
          );
          await act(() => {
            queryClient.setQueryData(
              savedWorkflowPickerQueryOptions({ page: 0, query: 'Previous' }).queryKey,
              {
                pageParams: [0, 1],
                pages: [
                  { items: [{ name: initialName, workflow_id: workflowId }], page: 0, pages: 2, total: 2 },
                  { items: [{ name: 'Other workflow', workflow_id: 'other' }], page: 1, pages: 2, total: 2 },
                ],
              },
              { updatedAt: 500 }
            );
          });
          expect(host.querySelector('[data-workflow-export-field-value="true"]')?.textContent).toBe(
            'Renamed child workflow'
          );
        }
        if (source === 'unavailable') {
          await act(() => {
            queryClient.setQueryData(savedWorkflowDetailQueryKey(workflowId), { name, workflow_id: workflowId });
          });
          expect(host.querySelector('[data-workflow-export-field-value="true"]')?.textContent).toBe(name);
        }
        expect(host.querySelector('.react-flow__node button')).toBeNull();
        expect(fetchSpy).not.toHaveBeenCalled();
        expect(adapter.commands.editGraph).not.toHaveBeenCalled();
      } finally {
        fetchSpy.mockRestore();
      }
    }
  );

  it.each([
    { cardinality: 'SINGLE', failed: false, teardown: false },
    { cardinality: 'COLLECTION', failed: false, teardown: false },
    { cardinality: 'SINGLE', failed: true, teardown: false },
    { cardinality: 'COLLECTION', failed: true, teardown: false },
    { cardinality: 'SINGLE', failed: false, teardown: true },
  ] as const)(
    'exports authored source images for $cardinality fields with failed=$failed and teardown=$teardown',
    async ({ cardinality, failed, teardown }) => {
      const source = outputImage(800, 400);
      const thumbnailSpy = vi.spyOn(galleryImageUrls, 'thumbnail').mockReturnValue(source);
      const value = { image_name: 'authored-source.png' };
      const imageTemplate: InvocationTemplate = {
        ...template,
        inputs: {
          a: {
            ...template.inputs.a!,
            input: 'any',
            title: 'Source Image',
            type: { batch: false, cardinality, name: 'ImageField' },
          },
        },
      };
      const imageNode: WorkflowInvocationNode = {
        ...documentNode,
        data: {
          ...documentNode.data,
          inputs: { a: { label: '', name: 'a', value: cardinality === 'SINGLE' ? value : [value] } },
        },
      };
      const graph = { ...projectGraph, nodes: [imageNode] };
      const execution = createExecutionPort();
      execution.set(completed(outputImage(100, 100)));
      const adapter = createAdapter(execution.port, projectSnapshotFor(graph));
      let capturedClone: HTMLElement | undefined;
      let exportedBlob: Blob | undefined;
      exportMocks.rasterize.mockImplementation(async (clone: HTMLElement, options: unknown) => {
        capturedClone = clone;
        const { rasterizeWorkflowImage } = await vi.importActual<{
          rasterizeWorkflowImage: typeof RasterizeWorkflowImage;
        }>('./workflowImageRaster');
        exportedBlob = await rasterizeWorkflowImage(clone, options as Parameters<typeof RasterizeWorkflowImage>[1]);
        return exportedBlob;
      });
      try {
        await render(adapter, 1, true, toFlowNodes(graph, [], { preview: imageTemplate }));
        const sourceImage = host.querySelector<HTMLImageElement>('[data-workflow-export-field-value="true"] img');
        expect(sourceImage).not.toBeNull();
        // Simulate a pending thumbnail's zero dimensions until decoding completes.
        const naturalWidthSpy = vi.spyOn(sourceImage!, 'naturalWidth', 'get').mockReturnValue(0);
        sourceImage!.style.width = '0px';
        sourceImage!.style.height = '0px';
        expect(sourceImage!.naturalWidth).toBe(0);
        const decode = sourceImage!.decode.bind(sourceImage);
        let releaseImage!: () => void;
        const imageReady = new Promise<void>((resolve) => {
          releaseImage = resolve;
        });
        const decodeSpy = vi.spyOn(sourceImage!, 'decode').mockImplementation(async () => {
          await imageReady;
          if (failed) {
            throw new Error('Thumbnail unavailable');
          }
          naturalWidthSpy.mockRestore();
          sourceImage!.style.removeProperty('width');
          sourceImage!.style.removeProperty('height');
          await decode();
        });
        expect(getComputedStyle(sourceImage!).objectFit).toBe(cardinality === 'SINGLE' ? 'contain' : 'cover');
        expect(sourceImage!.getBoundingClientRect().width).toBeLessThan(800);
        expect(sourceImage!.getBoundingClientRect().height).toBeLessThanOrEqual(128);
        const flowElement = host.querySelector<HTMLElement>('.react-flow')!;
        const exportPromise = exportWorkflowAsPng({
          bounds: { x: 20, y: 20, width: 300, height: 260 },
          fallbackWorkflowName: 'Untitled Workflow',
          flowElement,
          workflowName: 'Authored source image',
        });

        await vi.waitFor(() => expect(decodeSpy).toHaveBeenCalledOnce());
        expect(exportMocks.rasterize).not.toHaveBeenCalled();
        if (teardown) {
          await act(() => root.unmount());
          root = createRoot(host);
          const nextGraph = { ...projectGraph, id: 'next-workflow' };
          const nextAdapter = createAdapter(createExecutionPort().port, projectSnapshotFor(nextGraph));
          await render(nextAdapter);
          expect(flowElement.isConnected).toBe(false);
          expect(host.querySelector('.react-flow')?.isConnected).toBe(true);
        }
        releaseImage();
        const outcome = await exportPromise;
        decodeSpy.mockRestore();
        naturalWidthSpy.mockRestore();
        if (teardown) {
          expect(outcome).toEqual({ status: 'canceled' });
          expect(exportMocks.rasterize).not.toHaveBeenCalled();
          expect(downloadBlob).not.toHaveBeenCalled();
          return;
        }
        expect(outcome).toMatchObject({ status: 'exported', reduced: false });
        expect(exportedBlob?.type).toBe('image/png');
        const bitmap = await createImageBitmap(exportedBlob!);
        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        const context = canvas.getContext('2d')!;
        context.drawImage(bitmap, 0, 0);
        const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
        let sourcePixels = 0;
        for (let index = 0; index < pixels.length; index += 4) {
          if (pixels[index] === 76 && pixels[index + 1] === 139 && pixels[index + 2] === 245) {
            sourcePixels += 1;
          }
        }
        bitmap.close();
        if (failed) {
          expect(sourcePixels).toBe(0);
          expect(capturedClone?.querySelectorAll('img')).toHaveLength(0);
          expect(capturedClone?.querySelector('[data-workflow-export-field-value="true"] span')?.textContent).toBe(
            'authored-source.png'
          );
          // The fallback belongs only to the clone; the editor image is preserved for later loads.
          expect(host.querySelector('[data-workflow-export-field-value="true"] img')).toBe(sourceImage);
        } else {
          expect(sourcePixels).toBeGreaterThan(1000);
          expect(capturedClone?.querySelectorAll('img')).toHaveLength(1);
          expect(capturedClone?.querySelector('img')?.src).toBe(source);
        }
        expect(capturedClone?.textContent).toContain('authored-source.png');
        expect(capturedClone?.textContent).not.toContain('Latest output');
        expect(capturedClone?.querySelector('button, input')).toBeNull();
        expect(adapter.commands.editGraph).not.toHaveBeenCalled();
      } finally {
        thumbnailSpy.mockRestore();
      }
    }
  );

  it('shows running progress as the one ring in the status button, and omits it from the exported graph clone', async () => {
    const execution = createExecutionPort();
    const adapter = createAdapter(execution.port);
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter);
    await act(() =>
      execution.set({
        error: null,
        latestOutput: null,
        outputImageName: null,
        progress: 0.5,
        progressMessage: 'Denoising',
        status: 'running',
      })
    );
    const statusButton = host.querySelector<HTMLElement>('[data-node-status="running"]')!;
    const progressRings = host.querySelectorAll('.react-flow__node [data-scope="progress"][data-part="root"]');
    expect(statusButton.getAttribute('aria-label')).toBe('Status: Running (50%)');
    expect(progressRings).toHaveLength(1);
    expect(statusButton.contains(progressRings[0]!)).toBe(true);

    await act(() => userEvent.hover(statusButton));
    await vi.waitFor(() => expect(document.querySelector('[data-part="content"]')?.textContent).toContain('Denoising'));
    expect(document.querySelector('[data-part="content"]')?.textContent).toContain('Running (50%)');
    await act(() => userEvent.unhover(statusButton));

    await render(adapter, 1, true);

    const flowElement = host.querySelector<HTMLElement>('.react-flow')!;
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement,
      workflowName: 'Running workflow',
    });

    expect(capturedClone?.querySelector('[data-node-status-indicator="true"]')).toBeNull();
    expect(capturedClone?.querySelector('[data-scope="progress"]')).toBeNull();
  });

  it('outlines exported nodes in the theme border colour with the resting shadow', async () => {
    const adapter = createAdapter(createExecutionPort().port);
    let exported: { border: string; shadow: string } | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      const shell = getComputedStyle(clone.querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!);
      exported = { border: shell.borderColor, shadow: shell.boxShadow };
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true);
    const restingBorder = getComputedStyle(
      host.querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!
    ).borderColor;
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Outlined',
    });

    expect(exported?.border).toBe(restingBorder);
    expect(exported?.shadow).not.toBe('none');
  });

  it('omits field validation diagnostics from the static export', async () => {
    const requiredTemplate: InvocationTemplate = {
      ...template,
      inputs: { a: { ...template.inputs.a!, input: 'direct', required: true } },
    };
    const incompleteNode: WorkflowInvocationNode = {
      ...documentNode,
      data: { ...documentNode.data, inputs: { a: { label: 'A', name: 'a', value: undefined } } },
    };
    const nodes = toFlowNodes({ ...projectGraph, nodes: [incompleteNode] }, [], { preview: requiredTemplate });
    const adapter = createAdapter(createExecutionPort().port);
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, nodes);
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Incomplete workflow',
    });

    expect(capturedClone?.querySelector('[data-part="error-text"]')).toBeNull();
    expect(capturedClone?.querySelector('[aria-invalid="true"]')).toBeNull();
    expect(capturedClone?.textContent).not.toContain('Required value.');
  });

  it('renders stale record field values without picker diagnostics or controls in snapshot exports', async () => {
    const styleTemplate: InvocationTemplate = {
      ...template,
      inputs: {
        a: {
          ...template.inputs.a!,
          input: 'direct',
          title: 'Style Preset',
          type: { batch: false, cardinality: 'SINGLE', name: 'StylePresetField' },
        },
      },
    };
    const styleNode: WorkflowInvocationNode = {
      ...documentNode,
      data: {
        ...documentNode.data,
        inputs: { a: { label: 'Style Preset', name: 'a', value: { style_preset_id: 'missing-preset' } } },
        isOpen: true,
      },
    };
    const graph = { ...projectGraph, nodes: [styleNode] };
    const nodes = toFlowNodes(graph, [], { preview: styleTemplate });
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
    queryClient.setQueryData(['generation', 'promptTemplates', 'list'], []);
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, false, nodes);
    // The record lookup settles asynchronously; the default one-second wait is too short on a loaded machine.
    await vi.waitFor(() => expect(host.textContent).toContain(i18n.t('nodes.stylePresetMissing')), { timeout: 5_000 });
    await render(adapter, 1, true, nodes);
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Stale style preset',
    });

    expect(capturedClone?.textContent).toContain('missing-preset');
    expect(capturedClone?.textContent).not.toContain(i18n.t('nodes.stylePresetMissing'));
    expect(capturedClone?.querySelector('input')).toBeNull();
    expect(capturedClone?.querySelector('button')).toBeNull();
  });

  it.each([
    [0, '#ff000000'],
    [128, '#ff000080'],
    [255, '#ff0000ff'],
  ] as const)('preserves color alpha %s in snapshot exports', async (alpha, expected) => {
    const colorTemplate: InvocationTemplate = {
      ...template,
      inputs: {
        a: {
          ...template.inputs.a!,
          input: 'direct',
          title: 'Color',
          type: { batch: false, cardinality: 'SINGLE', name: 'ColorField' },
        },
      },
    };
    const colorNode: WorkflowInvocationNode = {
      ...documentNode,
      data: {
        ...documentNode.data,
        inputs: { a: { label: '', name: 'a', value: { r: 255, g: 0, b: 0, a: alpha } } },
      },
    };
    const graph = { ...projectGraph, nodes: [colorNode] };
    const nodes = toFlowNodes(graph, [], { preview: colorTemplate });
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, nodes);
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Color alpha',
    });

    const text = capturedClone?.querySelector('[data-workflow-export-field-value="true"]')?.textContent;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    context.fillStyle = text!;
    context.fillRect(0, 0, 1, 1);
    expect(context.getImageData(0, 0, 1, 1).data[3]).toBe(alpha);
    expect(text).toBe(expected);
  });

  it.each([
    ['fixed', '42'],
    ['random', 'Random'],
    ['increment', '42 (Increment)'],
    ['decrement', '42 (Decrement)'],
    [undefined, '42'],
  ] as const)('preserves authored seed mode %s in snapshot exports', async (seedMode, expected) => {
    const seedTemplate: InvocationTemplate = {
      ...template,
      inputs: {
        seed: { ...template.inputs.a!, input: 'direct', maximum: 4_294_967_295, name: 'seed', title: 'Seed' },
      },
    };
    const seedNode: WorkflowInvocationNode = {
      ...documentNode,
      data: { ...documentNode.data, inputs: { seed: { label: '', name: 'seed', seedMode, value: 42 } } },
    };
    const ordinaryNode: WorkflowInvocationNode = {
      ...documentNode,
      id: 'ordinary-integer',
      position: { x: 420, y: 20 },
      data: { ...documentNode.data, type: 'ordinary', inputs: { a: { label: '', name: 'a', seedMode, value: 42 } } },
    };
    const graph = { ...projectGraph, nodes: [seedNode, ordinaryNode] };
    const nodes = toFlowNodes(graph, [], {
      preview: seedTemplate,
      ordinary: { ...template, type: 'ordinary', inputs: { a: { ...template.inputs.a!, input: 'direct' } } },
    });
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, nodes);
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 720, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Seed modes',
    });

    expect(
      capturedClone?.querySelector(`[data-id="${NODE_ID}"] [data-workflow-export-field-value="true"]`)?.textContent
    ).toBe(expected);
    expect(
      capturedClone?.querySelector('[data-id="ordinary-integer"] [data-workflow-export-field-value="true"]')
        ?.textContent
    ).toBe('42');
    expect(capturedClone?.querySelector('input, button, [role="menu"]')).toBeNull();
  });

  it('omits legacy generator results while preserving generator settings in snapshot exports', async () => {
    const generatorTemplate: InvocationTemplate = {
      ...template,
      inputs: {
        a: {
          ...template.inputs.a!,
          input: 'direct',
          title: 'Values',
          type: { batch: false, cardinality: 'SINGLE', name: 'FloatGeneratorField' },
        },
      },
    };
    const generatorValue = {
      type: 'float_generator_arithmetic_sequence',
      start: 0,
      step: 0.2,
      count: 3,
      values: [0.2, 0.4, 0.6],
    };
    const generatorNode: WorkflowInvocationNode = {
      ...documentNode,
      data: {
        ...documentNode.data,
        inputs: { a: { label: 'Values', name: 'a', value: generatorValue } },
        isOpen: true,
      },
    };
    const graph = { ...projectGraph, nodes: [generatorNode] };
    const nodes = toFlowNodes(graph, [], { preview: generatorTemplate });
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, nodes);
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Generator workflow',
    });

    const fieldValue = capturedClone?.querySelector<HTMLElement>('[data-workflow-export-field-value="true"]');
    expect(fieldValue?.textContent).toBe(
      '{"type":"float_generator_arithmetic_sequence","start":0,"step":0.2,"count":3}'
    );
    expect(capturedClone?.textContent).not.toContain('0.2, 0.4, 0.6');
    expect(capturedClone?.querySelector('input')).toBeNull();
    expect(capturedClone?.querySelector('button')).toBeNull();
  });

  it('removes loop validation status from static boundary labels', async () => {
    const forNode: WorkflowInvocationNode = {
      ...documentNode,
      data: { ...documentNode.data, type: 'for', label: 'For' },
      id: 'for-node',
    };
    const bodyNode: WorkflowInvocationNode = {
      ...documentNode,
      data: { ...documentNode.data, type: 'number', label: 'Body' },
      id: 'body-node',
      position: { x: 300, y: 20 },
    };
    const returnNode: WorkflowInvocationNode = {
      ...documentNode,
      data: { ...documentNode.data, type: 'for_return', label: 'Return' },
      id: 'return-node',
      position: { x: 600, y: 20 },
    };
    const edges: WorkflowEdge[] = [
      {
        id: 'iteration-edge',
        source: forNode.id,
        sourceHandle: 'item',
        target: bodyNode.id,
        targetHandle: 'value',
        type: 'default',
      },
      {
        id: 'return-edge',
        source: bodyNode.id,
        sourceHandle: 'value',
        target: returnNode.id,
        targetHandle: 'output',
        type: 'default',
      },
    ];
    const graph = { ...projectGraph, edges, nodes: [forNode, bodyNode, returnNode] };
    const nodes = toFlowNodes(graph, [], {});
    const adapter = createAdapter(createExecutionPort().port);
    const renderBoundary = (isExporting: boolean) =>
      act(() =>
        root.render(
          <ChakraProvider value={system}>
            <WorkflowImageExportProvider isExporting={isExporting}>
              <WorkflowUiProvider adapter={adapter}>
                <ReactFlow edges={[]} nodes={nodes} nodeTypes={nodeTypes}>
                  <LoopBodyBoundaryOverlay edges={edges} nodes={graph.nodes} />
                </ReactFlow>
              </WorkflowUiProvider>
            </WorkflowImageExportProvider>
          </ChakraProvider>
        )
      );

    await renderBoundary(false);
    const editorBoundary = await vi.waitFor(() => {
      const boundary = host.querySelector<HTMLElement>('[data-loop-body-boundary="for-node"]');
      expect(boundary).not.toBeNull();
      return boundary!;
    });
    expect(editorBoundary.getAttribute('data-loop-body-status')).toBe('missing_linkage');
    expect(editorBoundary.getAttribute('aria-label')).toContain(
      i18n.t('nodes.forLoopBodyBoundaryStatus.missing_linkage')
    );

    await renderBoundary(true);
    const snapshotBoundary = await vi.waitFor(() => {
      const boundary = host.querySelector<HTMLElement>('[data-loop-body-boundary="for-node"]');
      expect(boundary).not.toBeNull();
      return boundary!;
    });
    expect(snapshotBoundary.getAttribute('aria-label')).toBe(i18n.t('nodes.forLoopBodyBoundary'));
    expect(snapshotBoundary.hasAttribute('data-loop-body-status')).toBe(false);
  });

  it('omits gallery images from static exports while the editor shows the live generation frame', async () => {
    const liveImageUrl = 'data:image/png;base64,bGl2ZQ==';
    const savedImageUrl = 'data:image/png;base64,c2F2ZWQ=';
    let capturedClone: HTMLElement | undefined;
    exportMocks.progressImage = { dataUrl: liveImageUrl, height: 2, width: 2 };
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
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
    expect(exportedImage).toBeNull();
    expect(capturedClone?.textContent).not.toContain('No image yet');
    expect(capturedClone?.textContent).not.toContain('generating');
  });

  it('omits failure status and backend diagnostics from the static export', async () => {
    const execution = createExecutionPort(callNodeId);
    execution.set({
      error: 'Child node failed',
      latestOutput: null,
      outputImageName: null,
      progress: null,
      progressMessage: null,
      status: 'failed',
    });
    const adapter = createAdapter(execution.port);
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, callFlowNodes);
    const flowElement = host.querySelector<HTMLElement>('.react-flow')!;
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 300, height: 260 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement,
      workflowName: 'Failed workflow',
    });

    expect(capturedClone?.querySelector('[data-node-status-indicator="true"]')).toBeNull();
    expect(capturedClone?.textContent).not.toContain('Failed');
    expect(capturedClone?.textContent).not.toContain('Child node failed');
    expect(capturedClone?.textContent).not.toContain('Child workflow error');
  });

  it('omits missing-template diagnostics from the static export', async () => {
    const unknownLabel = `Preserved unknown node ${'x'.repeat(240)}`;
    const unknownNode: WorkflowInvocationNode = {
      ...documentNode,
      data: { ...documentNode.data, label: unknownLabel, type: 'removed_invocation' },
      id: 'unknown-node',
      position: { x: 420, y: 20 },
    };
    const graph = { ...projectGraph, nodes: [documentNode, unknownNode] };
    const nodes = toFlowNodes(graph, [], { preview: template });
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, nodes);
    const flowElement = host.querySelector<HTMLElement>('.react-flow')!;
    const knownShell = host.querySelector<HTMLElement>(
      '.react-flow__node[data-id="preview-node"] [data-workflow-node-shell="true"]'
    );
    const unknownShell = host.querySelector<HTMLElement>(
      '.react-flow__node[data-id="unknown-node"] [data-workflow-node-shell="true"]'
    );
    const unknownTitle = unknownShell?.querySelector<HTMLElement>('p');
    expect(unknownShell?.textContent).toContain(unknownLabel);
    expect(unknownShell?.textContent).not.toContain('Unknown node type');
    expect(getComputedStyle(unknownShell!).borderColor).toBe(getComputedStyle(knownShell!).borderColor);
    expect(unknownTitle?.textContent).toBe(unknownLabel);
    expect(getComputedStyle(unknownTitle!).whiteSpace).toBe('nowrap');
    expect(unknownTitle?.dataset.workflowExportStaticNodeContent).toBe('true');

    const bounds = getWorkflowContentBounds(flowElement, { x: 20, y: 20, width: 600, height: 100 });
    expect(bounds.height).toBeCloseTo(
      Math.max(knownShell!.getBoundingClientRect().height, unknownShell!.getBoundingClientRect().height)
    );
    expect(bounds.height).toBeLessThan(100);

    await exportWorkflowAsPng({
      bounds,
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement,
      workflowName: 'Unknown node',
    });

    expect(capturedClone?.textContent).toContain(unknownLabel);
    expect(capturedClone?.textContent).not.toContain('Unknown node type');
  });

  it('removes editor selection styling from nodes and edges in the static export', async () => {
    const targetNode = { ...documentNode, id: 'target-node', position: { x: 420, y: 20 } };
    const graph: ProjectGraphState = {
      ...projectGraph,
      edges: [
        {
          id: 'edge',
          source: NODE_ID,
          sourceHandle: 'value',
          target: targetNode.id,
          targetHandle: 'a',
          type: 'default',
        },
      ],
      nodes: [documentNode, targetNode],
    };
    const nodes = toFlowNodes(graph, [], { preview: template }).map((node) => ({
      ...node,
      selected: node.id === NODE_ID,
    }));
    const edges = toFlowEdges(graph, [], 'default', new Set([NODE_ID]), { preview: template });
    const exportRef = createRef<HTMLDivElement>();
    const adapter = createAdapter(createExecutionPort().port, projectSnapshotFor(graph));

    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <WorkflowImageExportView containerRef={exportRef} edges={edges} nodes={nodes} nodeTypes={nodeTypes} />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );

    const selectedShell = exportRef.current?.querySelector<HTMLElement>(
      '.react-flow__node[data-id="preview-node"] [data-workflow-node-shell="true"]'
    );
    const plainShell = exportRef.current?.querySelector<HTMLElement>(
      '.react-flow__node[data-id="target-node"] [data-workflow-node-shell="true"]'
    );
    expect(selectedShell).not.toBeNull();
    expect(plainShell).not.toBeNull();
    expect(getComputedStyle(selectedShell!).boxShadow).toBe(getComputedStyle(plainShell!).boxShadow);
    expect(exportRef.current?.querySelector('.react-flow__edge.workflow-selected-node-edge')).toBeNull();
    expect(exportRef.current?.querySelector('.react-flow__edge.animated')).toBeNull();
  });

  it('keeps the visible collapsed node unchanged while an expanded offscreen snapshot is rasterizing', async () => {
    const node = { ...documentNode, data: { ...documentNode.data, isOpen: false, notes: 'Snapshot metadata' } };
    const directNode = {
      ...documentNode,
      data: { ...documentNode.data, inputs: { a: { label: 'A', name: 'a', value: 42 } } },
      id: 'direct-node',
      position: { x: 20, y: 100 },
    };
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
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
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
    expect(offscreenInput).toBeNull();
    expect(capturedClone?.textContent).not.toContain('Snapshot metadata');
    expect(capturedClone?.textContent).toContain('42');
    expect(capturedClone?.querySelector('input')).toBeNull();
    expect(capturedClone?.querySelector('button[aria-label="Collapse node"]')).toBeNull();
    finish(new Blob(['png'], { type: 'image/png' }));
    await capture;
  });

  it.each(['completed', 'running', 'failed'] as const)('omits execution results from a %s snapshot', async (status) => {
    const execution = createExecutionPort();
    const completed = {
      error: null,
      latestOutput: { value: 'previous result' },
      outputImageName: 'data:image/png;base64,cHJldmlvdXM=',
      progress: null,
      progressMessage: null,
      status: 'completed' as const,
    };
    execution.set(completed);
    const adapter = createAdapter(execution.port);
    await render(adapter, 1, true);
    expect(host.textContent).not.toContain('previous result');
    await act(() => execution.set({ ...completed, status }));
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
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
    expect(capturedClone?.textContent).not.toContain('Failed');
  });

  it('blocks a third capture after two timeouts and recovers when one rasterization settles', async () => {
    vi.useFakeTimers();
    exportMocks.rasterize.mockClear();
    const finishRasterizations: Array<(blob: Blob) => void> = [];
    const lateBlob = new Blob(['late'], { type: 'image/png' });
    let signalBothStarted: () => void = () => {};
    const bothStarted = new Promise<void>((resolve) => {
      signalBothStarted = resolve;
    });
    exportMocks.rasterize.mockImplementation(
      () =>
        new Promise<Blob>((resolve) => {
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
      expect(exportMocks.rasterize).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(WORKFLOW_EXPORT_TIMEOUT_MS);
      await Promise.all([firstTimedOut, secondTimedOut]);

      await expect(exportWorkflowAsPng(exportOptions)).resolves.toEqual({ status: 'busy' });
      expect(exportMocks.rasterize).toHaveBeenCalledTimes(2);

      finishRasterizations[0]!(lateBlob);
      await vi.advanceTimersByTimeAsync(0);
      exportMocks.rasterize.mockResolvedValueOnce(new Blob(['png'], { type: 'image/png' }));
      await exportWorkflowAsPng(exportOptions);
      expect(exportMocks.rasterize).toHaveBeenCalledTimes(3);

      finishRasterizations[1]!(lateBlob);
      await vi.advanceTimersByTimeAsync(0);
    } finally {
      finishRasterizations.forEach((finish) => finish(lateBlob));
      await vi.advanceTimersByTimeAsync(0);
      vi.useRealTimers();
    }
  });

  // Outputs are inspected in the side panel; a result growing the node would push it over the nodes below.
  it('keeps its size and shows no image when a run completes with an image output', async () => {
    const execution = createExecutionPort();
    const adapter = createAdapter(execution.port);

    await render(adapter);
    const idleHeight = nodeHeight();

    await act(() => execution.set(completed(outputImage(400, 100))));

    expect(image()).toBeNull();
    expect(nodeHeight()).toBe(idleHeight);
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
    expect(host.querySelector('.react-flow__node input')).toBeNull();
    expect(host.querySelector('[data-workflow-export-field-value="true"]')?.textContent).toBe('42');
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
      outputImageName: null,
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

    const failureButton = await vi.waitFor(() => {
      const button = host.querySelector<HTMLElement>('button[aria-label="Status: Failed"]');
      expect(button).not.toBeNull();
      return button!;
    });

    await act(() => userEvent.hover(failureButton));
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
  /** What the node's status button says; the reason for a flagged field lives there, not under the field. */
  const nodeStatus = () =>
    host.querySelector('[data-node-status-indicator="true"]')?.getAttribute('aria-label') ?? null;

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
    expect(nodeStatus()).toBeNull();

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
    expect(nodeStatus()).toBe('Status: Weight is required.');

    await keys('7{Home}-');
    await settle();
    expect(weight.value).toBe('-7');
    expect(fieldValue(store, 'weight')).toBe(-7);
    expect(nodeStatus()).toBeNull();

    await keys('{Home}{Delete}');
    await settle();
    expect(weight.value).toBe('7');
    expect(fieldValue(store, 'weight')).toBe(7);

    await act(() => weight.blur());
    await settle();
    expect(weight.value).toBe('7');
  });

  it('renames a field from its name on double-click; Escape keeps the name and the template title clears it', async () => {
    const { store } = await renderEntryNode();
    const fieldLabel = () => {
      const node = store.getSnapshot().nodes.find((candidate) => candidate.id === entryNode.id);
      return node?.type === 'invocation' ? node.data.inputs.steps?.label : undefined;
    };
    const stepsTitle = () =>
      Array.from(host.querySelectorAll<HTMLElement>('[data-node-input-field-title="true"]')).find((title) =>
        title.textContent?.startsWith(fieldLabel() || 'Steps')
      )!;
    const rename = async (sequence: string) => {
      await act(() => userEvent.dblClick(stepsTitle()));
      const input = host.querySelector<HTMLInputElement>('input[aria-label="Field label"]')!;
      expect(document.activeElement).toBe(input);
      await keys(sequence);
      await settle();
    };

    await rename('Iterations{Enter}');
    expect(fieldLabel()).toBe('Iterations');
    expect(stepsTitle().textContent).toBe('Iterations *');
    expect(host.querySelector('input[aria-label="Field label"]')).toBeNull();
    // Focus returns to the node, where editor shortcuts such as undo still apply.
    expect(document.activeElement?.classList.contains('react-flow__node')).toBe(true);

    await rename('Discarded{Escape}');
    expect(fieldLabel()).toBe('Iterations');

    await rename('Steps{Enter}');
    expect(fieldLabel()).toBe('');
    expect(stepsTitle().textContent).toBe('Steps *');
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
    expect(nodeStatus()).toBe('Status: Steps: Invalid value.');

    await keys('{Backspace}{Backspace}');
    await settle();
    expect(steps.value).toBe('20');
    expect(fieldValue(store, 'steps')).toBe(20);
    expect(steps.getAttribute('aria-invalid')).toBeNull();
    expect(nodeStatus()).toBeNull();
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
  const statusButton = () => host.querySelector('.react-flow__node [data-node-status-indicator="true"]');
  const borderColor = () => getComputedStyle(host.querySelector<HTMLElement>('.react-flow__node > div')!).borderColor;

  it('marks a node whose template moved on with a warning border and an update status, and leaves a current one alone', async () => {
    await render(flowNodes);
    expect(statusButton()).toBeNull();
    const currentBorder = borderColor();

    await render(toFlowNodes(projectGraph, [], { preview: { ...template, version: '1.1.0' } }));
    expect(statusButton()?.getAttribute('aria-label')).toBe('Status: Version 1.0.0 → 1.1.0: update available');
    expect(statusButton()?.getAttribute('data-node-status')).toBe('updatable');
    expect(borderColor()).not.toBe(currentBorder);

    await render(toFlowNodes(projectGraph, [], { preview: { ...template, version: '2.0.0' } }));
    expect(statusButton()?.getAttribute('aria-label')).toBe(
      'Status: Version 1.0.0 cannot be updated to 2.0.0; delete and re-add the node'
    );
    expect(statusButton()?.getAttribute('data-node-status')).toBe('versionMismatch');
  });

  it('waits neutrally for node definitions and warns of an unknown type only once they have loaded', async () => {
    await render(flowNodes);
    const knownBorder = borderColor();
    const nodeText = () => host.querySelector('.react-flow__node')!.textContent;

    for (const isCompact of [false, true]) {
      await render(toFlowNodes(projectGraph, [], undefined, undefined, isCompact, false, 'loading'));
      expect(borderColor()).toBe(knownBorder);
      expect(nodeText()).toContain('preview');
      expect(nodeText()).toContain('Loading node…');
      expect(nodeText()).not.toContain('Unknown node type');
    }

    await render(toFlowNodes(projectGraph, [], undefined, undefined, false, false, 'error'));
    expect(borderColor()).toBe(knownBorder);
    expect(nodeText()).toContain('Node definitions could not be loaded.');

    await render(toFlowNodes(projectGraph, [], undefined, undefined, false, false, 'loaded'));
    expect(borderColor()).not.toBe(knownBorder);
    expect(nodeText()).toContain('Unknown node type "preview". It cannot run on this backend.');
  });

  it('leads an unknown type to the Nodes Manager for an account that installs node packs, and to an admin otherwise', async () => {
    const missingNodes = toFlowNodes(projectGraph, [], undefined, undefined, false, false, 'loaded');
    const renderFor = (canManageNodes: boolean, openNodesManager = vi.fn()) =>
      act(() =>
        root.render(
          <ChakraProvider value={system}>
            <WorkflowUiProvider
              adapter={{
                ...createAdapter(createExecutionPort().port),
                capabilities: { getSnapshot: () => ({ canManageNodes, canUseCache: true }), subscribe: () => () => {} },
                openNodesManager,
              }}
            >
              <ReactFlow edges={[]} nodes={missingNodes} nodeTypes={nodeTypes} />
            </WorkflowUiProvider>
          </ChakraProvider>
        )
      );
    const openNodesManager = vi.fn();

    await renderFor(true, openNodesManager);
    const openButton = [...host.querySelectorAll('button')].find(
      (button) => button.textContent === 'Open Nodes Manager'
    );

    await userEvent.click(openButton!);
    expect(openNodesManager).toHaveBeenCalledOnce();

    await renderFor(false);
    expect(host.querySelector('.react-flow__node')!.textContent).toContain(
      'Ask an administrator to install the node pack that provides it.'
    );
    expect(host.querySelector('.react-flow__node button')).toBeNull();
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
    expect(host.textContent).not.toContain(i18n.t('nodes.useCache'));

    await render('None');
    expect(header().textContent).toContain('(no group)');
  });
});

const PROBED_COLORS = [
  'accent.fg',
  'accent.solid',
  'bg',
  'bg.muted',
  'border.error',
  'fg.error',
  'fg.warning',
  'fg',
] as const;

describe('InvocationFlowNode header, states and field rows', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.cssText = 'width: 1000px; height: 520px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  const requiredInput = (
    name: string,
    title: string,
    overrides: Partial<InvocationTemplate['inputs'][string]> = {}
  ) => ({
    ...template.inputs.a!,
    input: 'any' as const,
    name,
    required: true,
    title,
    ...overrides,
  });
  const stateTemplate: InvocationTemplate = {
    ...template,
    description: 'Adds two numbers.',
    inputs: {
      a: requiredInput('a', 'A'),
      b: requiredInput('b', 'B', { input: 'connection' }),
    },
  };
  const stateNode = (id: string, x: number, value: number | undefined): WorkflowInvocationNode => ({
    ...documentNode,
    data: { ...documentNode.data, inputs: { a: { label: '', name: 'a', value } } },
    id,
    position: { x, y: 20 },
  });
  /** Each node's run state, as the execution port reports it. */
  const createExecutionMap = (states: Record<string, WorkflowNodeExecutionState>) => ({
    get: (nodeId: string) => states[nodeId] ?? null,
    subscribe: () => () => {},
  });
  const render = async (
    nodes: ReturnType<typeof toFlowNodes>,
    execution: Pick<WorkflowUiAdapter['nodeExecution'], 'get' | 'subscribe'> = createExecutionMap({})
  ) => {
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <WorkflowUiProvider adapter={createAdapter(execution)}>
              <Box css={flowThemeCss} h="full" w="full">
                <ReactFlow edges={[]} nodes={nodes} nodeTypes={nodeTypes} />
              </Box>
            </WorkflowUiProvider>
            {PROBED_COLORS.map((token) => (
              <Box key={token} color={token} data-color-probe={token} />
            ))}
          </QueryClientProvider>
        </ChakraProvider>
      )
    );
    // xyflow keeps a node `visibility: hidden` until it has measured it. Its box already has a size, but hit-tests and
    // pointer events pass through it, so a test that measures or points at a node waits for every node to show.
    await vi.waitFor(() =>
      expect(
        [...host.querySelectorAll<HTMLElement>('.react-flow__node')].every(
          (element) => getComputedStyle(element).visibility === 'visible'
        )
      ).toBe(true)
    );
  };
  const node = (id: string) => host.querySelector<HTMLElement>(`.react-flow__node[data-id="${id}"]`)!;
  const shell = (id: string) => node(id).querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!;
  const header = (id: string) => shell(id).firstElementChild as HTMLElement;
  const statusButton = (id: string) => node(id).querySelector<HTMLElement>('[data-node-status-indicator="true"]');
  const panelText = () => document.querySelector('[data-scope="tooltip"][data-part="content"]')?.textContent ?? '';
  /** A theme colour as computed styles report it, independent of how the node composes it. */
  const themeColor = (token: (typeof PROBED_COLORS)[number]) =>
    getComputedStyle(host.querySelector<HTMLElement>(`[data-color-probe="${token}"]`)!).color;
  const failed: WorkflowNodeExecutionState = { ...completed(''), error: 'CUDA out of memory', status: 'failed' };

  it('explains the node in its info panel on hover and on keyboard focus, and shows no status at rest', async () => {
    const labelled = {
      ...stateNode('info-node', 20, 4),
      data: { ...stateNode('info-node', 20, 4).data, label: 'Sum', notes: 'Feeds the seed.' },
    };
    const infoTemplate = { ...stateTemplate, inputs: { a: stateTemplate.inputs.a! } };
    await render(toFlowNodes({ ...projectGraph, nodes: [labelled] }, [], { preview: infoTemplate }));

    expect(statusButton('info-node')).toBeNull();
    const infoButton = node('info-node').querySelector<HTMLButtonElement>('button[aria-label="Show details for Sum"]')!;
    expect(document.querySelector('[data-scope="tooltip"][data-part="content"]')).toBeNull();

    await act(() => userEvent.hover(infoButton));
    await vi.waitFor(() => expect(panelText()).toContain('Sum (Preview)'));
    for (const line of [
      'Type: preview',
      'Node pack: invokeai',
      'Version: 1.0.0',
      'Adds two numbers.',
      'Feeds the seed.',
    ]) {
      expect(panelText()).toContain(line);
    }
    await act(() => userEvent.unhover(infoButton));
    await vi.waitFor(() => expect(panelText()).toBe(''));

    // Tab from the collapse button: the panel opens for keyboard focus too.
    await act(async () => {
      node('info-node').querySelector<HTMLButtonElement>('button[aria-label="Collapse node"]')!.focus();
      await userEvent.tab();
    });
    expect(document.activeElement).toBe(infoButton);
    await vi.waitFor(() => expect(panelText()).toContain('Sum (Preview)'));
  });

  it('says in the status panel what each blocking field needs, and asks for a model by name', async () => {
    await render(
      toFlowNodes({ ...projectGraph, nodes: [stateNode('invalid-node', 20, undefined)] }, [], {
        preview: stateTemplate,
      })
    );

    const button = statusButton('invalid-node')!;
    expect(button.getAttribute('aria-label')).toBe('Status: 2 fields need attention');
    await act(() => userEvent.hover(button));
    await vi.waitFor(() => expect(panelText()).toContain('A is required.'));
    expect(panelText()).toContain('B needs a connection.');
    expect(panelText()).not.toContain('Required value.');
    await act(() => userEvent.unhover(button));

    const modelTemplate: InvocationTemplate = {
      ...template,
      inputs: {
        model: requiredInput('model', 'Model', {
          input: 'direct',
          type: { batch: false, cardinality: 'SINGLE', name: 'ModelIdentifierField' },
        }),
      },
    };
    // A compact node draws no field controls but still reports what its fields need.
    await render(
      toFlowNodes(
        { ...projectGraph, nodes: [stateNode('model-node', 20, undefined)] },
        [],
        { preview: modelTemplate },
        undefined,
        true
      )
    );
    expect(statusButton('model-node')?.getAttribute('aria-label')).toBe('Status: Model is required. Pick a model.');
  });

  it('draws a failed node like an invalid one and leaves a completed node’s outline alone', async () => {
    const graph = {
      ...projectGraph,
      nodes: [
        stateNode('idle-node', 20, 1),
        stateNode('invalid-node', 340, undefined),
        stateNode('failed-node', 660, 1),
        { ...stateNode('completed-node', 20, 1), position: { x: 20, y: 260 } },
      ],
    };
    const connected = toFlowNodes(graph, [], { preview: stateTemplate }).map((flowNode) => ({
      ...flowNode,
      data: { ...flowNode.data, connectedTargetHandles: flowNode.id === 'invalid-node' ? [] : ['b'] },
    })) as ReturnType<typeof toFlowNodes>;
    await render(connected, createExecutionMap({ 'completed-node': completed(''), 'failed-node': failed }));

    const look = (id: string) => ({
      border: getComputedStyle(shell(id)).borderColor,
      headerBackground: getComputedStyle(header(id)).backgroundColor,
      headerColor: getComputedStyle(header(id)).color,
    });
    expect(look('failed-node')).toEqual(look('invalid-node'));
    expect(look('invalid-node').border).toBe(themeColor('border.error'));
    expect(look('invalid-node').headerBackground).not.toBe(look('idle-node').headerBackground);
    expect(look('completed-node')).toEqual(look('idle-node'));

    expect(statusButton('idle-node')).toBeNull();
    expect(statusButton('completed-node')?.getAttribute('aria-label')).toBe('Status: Completed');
    const failedButton = statusButton('failed-node')!;
    expect(failedButton.getAttribute('aria-label')).toBe('Status: Failed');
    // No status glyph is a circle, so none reads as the circled "i" of the info button beside it.
    const glyph = (button: HTMLElement) => button.querySelector('svg')!.getAttribute('class');
    expect(glyph(statusButton('invalid-node')!)).toContain('lucide-triangle-alert');
    expect(glyph(failedButton)).toContain('lucide-octagon-x');
    expect(glyph(statusButton('completed-node')!)).toContain('lucide-check');
    expect(glyph(node('failed-node').querySelector<HTMLElement>('[data-node-info-icon="true"]')!)).toContain(
      'lucide-info'
    );
    await act(() => userEvent.hover(failedButton));
    await vi.waitFor(() => expect(panelText()).toContain('CUDA out of memory'));
  });

  it('rings a selected node in its state colour, a focused node in a ring of its own, and outlines it on hover', async () => {
    const outdatedNode = {
      ...stateNode('outdated-node', 660, 1),
      data: { ...stateNode('o', 0, 1).data, type: 'outdated' },
    };
    const graph = {
      ...projectGraph,
      nodes: [stateNode('valid-node', 20, 1), stateNode('invalid-node', 340, undefined), outdatedNode],
    };
    const templates = { outdated: { ...stateTemplate, type: 'outdated', version: '1.1.0' }, preview: stateTemplate };
    const nodes = toFlowNodes(graph, [], templates).map((flowNode) => ({
      ...flowNode,
      data: { ...flowNode.data, connectedTargetHandles: flowNode.id === 'invalid-node' ? [] : ['b'] },
      selected: true,
    })) as ReturnType<typeof toFlowNodes>;
    await render(nodes);

    const shadow = (id: string) => getComputedStyle(shell(id)).boxShadow;
    expect(shadow('valid-node')).toContain(themeColor('accent.solid'));
    expect(shadow('invalid-node')).toContain(themeColor('border.error'));
    expect(shadow('invalid-node')).not.toContain(themeColor('accent.solid'));
    expect(shadow('outdated-node')).toContain(themeColor('fg.warning'));
    expect(getComputedStyle(shell('outdated-node')).borderColor).toBe(themeColor('fg.warning'));
    expect(shadow('outdated-node')).not.toContain(themeColor('accent.solid'));
    expect(shadow('valid-node')).not.toContain(themeColor('fg'));

    await act(async () => {
      await userEvent.keyboard('{Shift}');
      node('invalid-node').focus();
    });
    await vi.waitFor(() => expect(shadow('invalid-node')).toContain(themeColor('fg')));
    expect(shadow('invalid-node')).toContain(themeColor('border.error'));

    await render(toFlowNodes({ ...projectGraph, nodes: [stateNode('valid-node', 20, 1)] }, [], { preview: template }));
    const restingBorder = getComputedStyle(shell('valid-node')).borderColor;
    await act(() => userEvent.hover(shell('valid-node')));
    await vi.waitFor(() => expect(getComputedStyle(shell('valid-node')).borderColor).toBe(themeColor('accent.solid')));
    expect(restingBorder).not.toBe(themeColor('accent.solid'));
  });

  it('keeps the space where a node title or an output title is split, as an untrimmed label draws it', async () => {
    const splitTemplate = {
      ...stateTemplate,
      outputs: { value: { ...stateTemplate.outputs.value!, title: 'Image to Latents' } },
      title: 'Latents to Image',
    };
    await render(
      toFlowNodes({ ...projectGraph, nodes: [stateNode('split-node', 20, 1)] }, [], { preview: splitTemplate })
    );

    for (const text of ['Latents to Image', 'Image to Latents']) {
      const label = node('split-node').querySelector<HTMLElement>(`[data-full-text="${text}"]`)!;
      const [head, tail] = [...label.children] as HTMLElement[];
      expect(getComputedStyle(head!).whiteSpace).toBe('pre');
      // The same label untrimmed, beside it in the same font: its spans hold their own white-space.
      const reference = label.cloneNode(true) as HTMLElement;
      reference.style.cssText = 'position: absolute; visibility: hidden; overflow: visible';
      for (const span of reference.children as HTMLCollectionOf<HTMLElement>) {
        span.style.cssText = 'text-box: normal; white-space: pre; flex-shrink: 0';
      }
      label.parentElement!.append(reference);
      const width = [...reference.children].reduce((sum, span) => sum + span.getBoundingClientRect().width, 0);
      reference.remove();
      expect(head!.getBoundingClientRect().width + tail!.getBoundingClientRect().width).toBeCloseTo(width, 0);
    }
  });

  it('opens rename inputs over their label without moving anything around them', async () => {
    await render(rowNodes(4));
    const nodeRect = () => node('row-node').getBoundingClientRect();
    const buttonRects = () =>
      [...header('row-node').querySelectorAll('button')].map((button) => button.getBoundingClientRect().toJSON());
    const rowRect = () => fieldTitle('A').closest('[data-node-field-row]')!.getBoundingClientRect();
    const before = { buttons: buttonRects(), node: nodeRect().height, row: rowRect().height };
    const titleFontSize = parseFloat(getComputedStyle(header('row-node').querySelector('[title]')!).fontSize);
    const labelFontSize = parseFloat(getComputedStyle(fieldTitle('A')).fontSize);

    await act(() => userEvent.dblClick(header('row-node').querySelector<HTMLElement>('[title]')!));
    const titleInput = host.querySelector<HTMLInputElement>('input[aria-label="Node label"]')!;
    expect(document.activeElement).toBe(titleInput);
    // Each rename input keeps the text size of the label it replaces.
    expect(parseFloat(getComputedStyle(titleInput).fontSize)).toBe(titleFontSize);
    expect(nodeRect().height).toBe(before.node);
    expect(buttonRects()).toEqual(before.buttons);
    // Its own surface, not the header's tint.
    expect(getComputedStyle(titleInput).backgroundColor).not.toBe(getComputedStyle(header('row-node')).backgroundColor);
    await act(() => userEvent.keyboard('{Escape}'));

    await act(() => userEvent.dblClick(fieldTitle('A')));
    const labelInput = host.querySelector<HTMLInputElement>('input[aria-label="Field label"]')!;
    expect(parseFloat(getComputedStyle(labelInput).fontSize)).toBe(labelFontSize);
    expect(labelInput.getBoundingClientRect().right).toBeLessThan(nodeRect().right);
    expect(rowRect().height).toBe(before.row);
    expect(nodeRect().height).toBe(before.node);
    await act(() => userEvent.keyboard('{Escape}'));
    expect(nodeRect().height).toBe(before.node);
  });

  it('sets the header’s end buttons as far from the right border as the collapse button is from the left', async () => {
    const graph = {
      ...projectGraph,
      nodes: [stateNode('valid-node', 20, 1), stateNode('invalid-node', 340, undefined)],
    };
    await render(
      toFlowNodes(graph, [], { preview: stateTemplate }).map((flowNode) => ({
        ...flowNode,
        data: { ...flowNode.data, connectedTargetHandles: flowNode.id === 'valid-node' ? ['b'] : [] },
      })) as ReturnType<typeof toFlowNodes>
    );

    for (const id of ['valid-node', 'invalid-node']) {
      const shellRect = shell(id).getBoundingClientRect();
      const border = parseFloat(getComputedStyle(shell(id)).borderLeftWidth);
      const buttons = [...header(id).querySelectorAll<HTMLElement>('button')].map((button) =>
        button.getBoundingClientRect()
      );
      const startInset = buttons[0]!.left - (shellRect.left + border);
      const endInset = shellRect.right - border - buttons.at(-1)!.right;
      expect(startInset).toBeGreaterThan(0);
      expect(endInset).toBeCloseTo(startInset, 1);
    }
  });

  it('strengthens a red or amber outline on hover instead of leaving it unchanged', async () => {
    const outdatedNode = {
      ...stateNode('outdated-node', 340, 1),
      data: { ...stateNode('o', 0, 1).data, type: 'outdated' },
    };
    const graph = { ...projectGraph, nodes: [stateNode('invalid-node', 20, undefined), outdatedNode] };
    const templates = { outdated: { ...stateTemplate, type: 'outdated', version: '1.1.0' }, preview: stateTemplate };
    await render(
      toFlowNodes(graph, [], templates).map((flowNode) => ({
        ...flowNode,
        data: { ...flowNode.data, connectedTargetHandles: ['b'] },
      })) as ReturnType<typeof toFlowNodes>
    );

    for (const [id, resting] of [
      ['invalid-node', themeColor('border.error')],
      ['outdated-node', themeColor('fg.warning')],
    ] as const) {
      expect(getComputedStyle(shell(id)).borderColor).toBe(resting);
      await act(() => userEvent.hover(header(id)));
      await vi.waitFor(() => {
        const hovered = getComputedStyle(shell(id)).borderColor;
        expect(hovered).not.toBe(resting);
        expect(hovered).not.toBe(themeColor('accent.solid'));
      });
      await act(() => userEvent.unhover(header(id)));
    }
  });

  /** A node with one editable field `A` (connectable) and one connection-only field `B`, fed by a connection. */
  const rowNodes = (value: number | undefined, exposed: string[] = []) =>
    toFlowNodes({ ...projectGraph, nodes: [stateNode('row-node', 20, value)] }, [], { preview: stateTemplate }).map(
      (flowNode) => ({
        ...flowNode,
        data: { ...flowNode.data, connectedTargetHandles: ['b'], exposedFieldNames: exposed },
      })
    ) as ReturnType<typeof toFlowNodes>;
  const fieldTitle = (name: string) =>
    [...node('row-node').querySelectorAll<HTMLElement>('[data-node-input-field-title="true"]')].find((title) =>
      title.textContent?.startsWith(name)
    )!;
  const rowActions = (name: string) =>
    fieldTitle(name)
      .closest('[data-scope="field"][data-part="root"]')!
      .querySelector<HTMLElement>('[data-node-field-actions]')!;
  const opacity = (element: HTMLElement) => getComputedStyle(element).opacity;

  it('shows a row’s actions in legacy order on hover and on keyboard focus, and keeps them operable by keyboard', async () => {
    const adapter = createAdapter(createExecutionMap({}));
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <WorkflowUiProvider adapter={adapter}>
              <Box css={flowThemeCss} h="full" w="full">
                <ReactFlow edges={[]} nodes={rowNodes(4)} nodeTypes={nodeTypes} />
              </Box>
            </WorkflowUiProvider>
          </QueryClientProvider>
        </ChakraProvider>
      )
    );

    const actions = rowActions('A');
    expect([...actions.querySelectorAll('button')].map((button) => button.getAttribute('aria-label'))).toEqual([
      'Edit field description',
      'Reset A to default value',
      'Add A to the form',
    ]);
    // A connection-fed row keeps only the description: there is no value to reset or put in the form.
    expect([...rowActions('B').querySelectorAll('button')].map((button) => button.getAttribute('aria-label'))).toEqual([
      'Edit field description',
    ]);
    expect(opacity(actions)).toBe('0');

    await act(() => userEvent.hover(fieldTitle('A')));
    await vi.waitFor(() => expect(opacity(actions)).toBe('1'));
    expect(opacity(rowActions('B'))).toBe('0');
    await act(() => userEvent.unhover(fieldTitle('A')));
    await vi.waitFor(() => expect(opacity(actions)).toBe('0'));

    // Keyboard: focus in the row's control shows the actions, and Tab reaches them.
    const control = node('row-node').querySelector<HTMLInputElement>('input[aria-label="A"]')!;
    await act(async () => {
      control.focus();
      await userEvent.keyboard('{Shift>}{Tab}{/Shift}');
    });
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Add A to the form');
    await vi.waitFor(() => expect(opacity(actions)).toBe('1'));
    await act(() => userEvent.keyboard('{Enter}'));
    expect(adapter.commands.editGraph).toHaveBeenCalledWith({
      fieldIdentifier: { fieldName: 'a', nodeId: 'row-node' },
      type: 'exposeField',
    });
  });

  it('marks a field in the form on its label, blue with a link, instead of a lit button', async () => {
    await render(rowNodes(4, ['a']));

    const title = fieldTitle('A');
    expect(getComputedStyle(title).color).toBe(themeColor('accent.fg'));
    expect(title.querySelector('[role="img"][aria-label="In the form"]')).not.toBeNull();
    expect(getComputedStyle(fieldTitle('B')).color).not.toBe(themeColor('accent.fg'));
    const formButton = rowActions('A').querySelector<HTMLElement>('button[aria-label="Remove A from the form"]')!;
    expect(getComputedStyle(formButton).color).not.toBe(themeColor('accent.solid'));
  });

  it('marks an invalid field on its label and control border, with the reason on hover rather than under the field', async () => {
    await render(rowNodes(undefined));

    const title = fieldTitle('A');
    const control = node('row-node').querySelector<HTMLInputElement>('input[aria-label="A"]')!;
    expect(getComputedStyle(title).color).toBe(themeColor('fg.error'));
    expect(control.getAttribute('aria-invalid')).toBe('true');
    expect(getComputedStyle(control).borderColor).toBe(themeColor('border.error'));
    // The value sits in the node's recessed well, a step darker than the body, and keeps its error border on it.
    expect(getComputedStyle(control).backgroundColor).toBe(themeColor('bg'));
    expect(themeColor('bg')).not.toBe(themeColor('bg.muted'));
    expect(node('row-node').textContent).not.toContain('Required value.');
    // The reason is read with the control but not drawn under it.
    const description = node('row-node').querySelector<HTMLElement>('[data-part="error-text"]')!;
    expect(description.textContent).toBe('A is required.');
    // The field links it as the control's error message, as it does in the form view.
    expect(control.getAttribute('aria-errormessage')).toBe(description.id);
    expect(description.getBoundingClientRect().height).toBeLessThanOrEqual(1);

    await act(() => userEvent.hover(title));
    await vi.waitFor(() => expect(panelText()).toContain('A is required.'));
  });

  it('sets a last control as far from the node’s bottom as from its sides, and leaves a last label row tight', async () => {
    const controlLast = { ...stateTemplate, inputs: { a: stateTemplate.inputs.a! } };
    await render(toFlowNodes({ ...projectGraph, nodes: [stateNode('row-node', 20, 4)] }, [], { preview: controlLast }));

    const shellElement = node('row-node').querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!;
    const inner = shellElement.getBoundingClientRect();
    const border = parseFloat(getComputedStyle(shellElement).borderBottomWidth);
    const control = node('row-node').querySelector<HTMLElement>('[data-node-field-control]')!.getBoundingClientRect();
    const sideGap = control.left - (inner.left + border);
    const bottomGap = inner.bottom - border - control.bottom;
    expect(sideGap).toBeGreaterThan(8);
    expect(bottomGap).toBeCloseTo(sideGap, 0);

    // Ending in B, a row with no control, keeps the rows' own bottom padding.
    await render(rowNodes(4));
    const lastRow = fieldTitle('B').closest<HTMLElement>('[data-node-field-row]')!;
    const firstRow = fieldTitle('A').closest<HTMLElement>('[data-node-field-row]')!;
    expect(getComputedStyle(lastRow).paddingBottom).toBe(getComputedStyle(firstRow).paddingBottom);
  });

  it('fills a text area’s faded bottom with the same well as the text area itself', async () => {
    const promptInput = requiredInput('a', 'A', {
      input: 'direct',
      type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
      uiComponent: 'textarea',
    });
    await render(
      toFlowNodes(
        {
          ...projectGraph,
          nodes: [
            {
              ...stateNode('row-node', 20, undefined),
              data: {
                ...stateNode('row-node', 20, undefined).data,
                inputs: { a: { label: '', name: 'a', value: 'text' } },
              },
            },
          ],
        },
        [],
        { preview: { ...stateTemplate, inputs: { a: promptInput } } }
      )
    );

    const textarea = node('row-node').querySelector<HTMLTextAreaElement>('textarea')!;
    const fade = node('row-node').querySelector<HTMLElement>('[data-resizable-textarea-fade]')!;
    const fadeEnd = [...getComputedStyle(fade).backgroundImage.matchAll(/(?:rgba?|oklch)\([^)]*\)/g)].at(-1)?.[0];
    expect(getComputedStyle(textarea).backgroundColor).toBe(themeColor('bg'));
    expect(getComputedStyle(fade).display).not.toBe('none');
    expect(fadeEnd).toBe(getComputedStyle(textarea).backgroundColor);
  });

  it('gives each port a 24×24 grab area centred on its 12px dot', async () => {
    await render(rowNodes(4));

    const handle = node('row-node').querySelector<HTMLElement>('.react-flow__handle[data-handleid="a"]')!;
    const dot = handle.getBoundingClientRect();
    const centre = { x: dot.left + dot.width / 2, y: dot.top + dot.height / 2 };
    expect(dot.width).toBe(12);
    // 10px out is past the dot but inside the grab area; 14px out is past both.
    expect(document.elementFromPoint(centre.x - 10, centre.y + 10)).toBe(handle);
    expect(document.elementFromPoint(centre.x - 14, centre.y)).not.toBe(handle);
  });

  it('keeps a batch port’s grab area an upright 24×24 square, not its dot’s diamond', async () => {
    const batchInput = requiredInput('a', 'A', {
      input: 'connection',
      type: { batch: true, cardinality: 'COLLECTION', name: 'IntegerField' },
    });
    await render(
      toFlowNodes({ ...projectGraph, nodes: [stateNode('row-node', 20, 4)] }, [], {
        preview: { ...stateTemplate, inputs: { a: batchInput } },
      })
    );

    const handle = node('row-node').querySelector<HTMLElement>('.react-flow__handle[data-handleid="a"]')!;
    expect(handle.style.transform).toContain('rotate(45deg)');
    const dot = handle.getBoundingClientRect();
    const centre = { x: dot.left + dot.width / 2, y: dot.top + dot.height / 2 };
    // A corner of the upright square is inside; a diamond's tip, 16px straight out, is not.
    expect(document.elementFromPoint(centre.x - 11, centre.y - 11)).toBe(handle);
    expect(document.elementFromPoint(centre.x, centre.y + 16)).not.toBe(handle);
  });

  it('reports a HiDiffusion ratio order problem once while marking both ratios', async () => {
    const denoiseTemplates = parseOpenApiToTemplates(openapi);
    const denoise = buildInvocationNode(denoiseTemplates.denoise_latents!, { x: 0, y: 0 });
    denoise.data.inputs.hidiffusion!.value = true;
    denoise.data.inputs.hidiffusion_auto_ratios!.value = false;
    denoise.data.inputs.hidiffusion_t1_ratio!.value = 0.2;
    denoise.data.inputs.hidiffusion_t2_ratio!.value = 0.7;
    await render(toFlowNodes({ ...projectGraph, nodes: [denoise] }, [], denoiseTemplates));

    for (const name of ['hidiffusion_t1_ratio', 'hidiffusion_t2_ratio']) {
      expect(
        host.querySelector(`input[id="${denoise.id}-${name}-value-number-input"]`)?.getAttribute('aria-invalid')
      ).toBe('true');
    }
    await act(() => userEvent.hover(host.querySelector<HTMLElement>('[data-node-status-indicator="true"]')!));
    await vi.waitFor(() => expect(panelText()).toContain('T2 ratio must be less'));
    expect(panelText().split('T2 ratio must be less')).toHaveLength(2);
  });
});

describe('InvocationFlowNode collapsed ports', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.cssText = 'width: 1000px; height: 520px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  const placed = (id: string, x: number, y: number, isOpen: boolean): WorkflowInvocationNode => ({
    ...documentNode,
    data: { ...documentNode.data, isOpen },
    id,
    position: { x, y },
  });
  const edge = (id: string, source: string, target: string): WorkflowEdge => ({
    id,
    source,
    sourceHandle: 'value',
    target,
    targetHandle: 'a',
    type: 'default',
  });
  // a and b collapse with two edges between them; c stays open and is fed by a.
  const graphWith = (isOpen: boolean): ProjectGraphState => ({
    ...projectGraph,
    edges: [edge('ab1', 'a', 'b'), edge('ab2', 'a', 'b'), edge('ac', 'a', 'c')],
    nodes: [placed('a', 20, 20, isOpen), placed('b', 420, 20, isOpen), placed('c', 420, 260, true)],
  });
  const render = (graph: ProjectGraphState) =>
    act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={createAdapter(createExecutionPort().port, projectSnapshotFor(graph))}>
            <Box css={flowThemeCss} h="full" w="full">
              <ReactFlow
                edges={toFlowEdges(graph, [], 'default', undefined, templates)}
                edgeTypes={{ default: WorkflowEdgeComponent }}
                nodes={toFlowNodes(graph, [], templates)}
                nodeTypes={nodeTypes}
              />
            </Box>
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );
  const node = (id: string) => host.querySelector<HTMLElement>(`.react-flow__node[data-id="${id}"]`)!;
  const drawnEdgeIds = () =>
    [...host.querySelectorAll<SVGElement>('.react-flow__edge')].map((element) => element.getAttribute('data-id'));
  const centre = (rect: DOMRect) => ({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });

  it('merges a collapsed node’s ports, draws its edges from them with a count, and restores them on expand', async () => {
    await render(graphWith(false));

    const sourcePort = node('a').querySelector<HTMLElement>('[data-collapsed-port="right"]')!;
    expect(node('b').querySelector('[data-collapsed-port="left"]')).not.toBeNull();
    await vi.waitFor(() => expect(drawnEdgeIds().sort()).toEqual(['ab1', 'ac']));
    expect(host.querySelector('[data-collapsed-edge-count]')?.getAttribute('data-collapsed-edge-count')).toBe('2');

    // The edge to the open node leaves the merged port's outer side.
    const path = host.querySelector<SVGPathElement>('.react-flow__edge[data-id="ac"] .react-flow__edge-path')!;
    const start = path.getPointAtLength(0).matrixTransform(path.getScreenCTM()!);
    const port = sourcePort.getBoundingClientRect();
    expect(Math.abs(start.x - port.right)).toBeLessThan(1);
    expect(Math.abs(start.y - centre(port).y)).toBeLessThan(1);

    // The field handles behind the merged port take no new connections; legacy asked for the node to be opened.
    const hiddenHandle = node('a').querySelector<HTMLElement>('.react-flow__handle[data-handleid="value"]')!;
    expect(getComputedStyle(hiddenHandle).visibility).toBe('hidden');
    expect(hiddenHandle.classList.contains('connectable')).toBe(false);

    await render(graphWith(true));
    await vi.waitFor(() => expect(drawnEdgeIds().sort()).toEqual(['ab1', 'ab2', 'ac']));
    expect(host.querySelector('[data-collapsed-port]')).toBeNull();
    expect(host.querySelector('[data-collapsed-edge-count]')).toBeNull();
  });

  /**
   * The middle of a label's lowercase letters, which is what a reader lines a port up with, from the font itself: a 1ex
   * inline box standing on the label's baseline spans exactly its x-height.
   */
  const letterBodyCentre = (label: HTMLElement) => {
    const text = [label, ...label.querySelectorAll<HTMLElement>('*')]
      .reverse()
      .find((element) =>
        [...element.childNodes].some((child) => child.nodeType === Node.TEXT_NODE && child.textContent?.trim())
      )!;
    const probe = document.createElement('span');
    probe.style.cssText = 'display: inline-block; height: 1ex; vertical-align: baseline; width: 0';
    text.append(probe);
    const box = probe.getBoundingClientRect();
    probe.remove();
    return box.top + box.height / 2;
  };

  it('centres each port on its label’s letters, edges on the ports, and a collapsed node’s merged ports on its title', async () => {
    await render(graphWith(true));
    await vi.waitFor(() => expect(drawnEdgeIds()).toContain('ac'));

    const target = node('c').querySelector<HTMLElement>('.react-flow__handle[data-handleid="a"]')!;
    const source = node('a').querySelector<HTMLElement>('.react-flow__handle[data-handleid="value"]')!;
    const rows = [
      {
        handle: target,
        label: target.parentElement!.querySelector<HTMLElement>('[data-node-input-field-title="true"]')!,
      },
      {
        handle: source,
        label: source.parentElement!.querySelector<HTMLElement>('[data-node-output-field-title="true"]')!,
      },
    ];
    for (const { handle, label } of rows) {
      expect(Math.abs(centre(handle.getBoundingClientRect()).y - letterBodyCentre(label))).toBeLessThanOrEqual(0.5);
    }

    const path = host.querySelector<SVGPathElement>('.react-flow__edge[data-id="ac"] .react-flow__edge-path')!;
    const start = path.getPointAtLength(0).matrixTransform(path.getScreenCTM()!);
    const end = path.getPointAtLength(path.getTotalLength()).matrixTransform(path.getScreenCTM()!);
    expect(Math.abs(start.y - centre(source.getBoundingClientRect()).y)).toBeLessThan(1);
    expect(Math.abs(end.y - centre(target.getBoundingClientRect()).y)).toBeLessThan(1);

    await render(graphWith(false));
    const header = node('a').querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!.firstElementChild!;
    const title = [...header.querySelectorAll<HTMLElement>('p, span')].find(
      (element) => element.childElementCount === 0 && element.textContent === template.title
    )!;
    for (const side of ['left', 'right']) {
      const port = node('a').querySelector<HTMLElement>(`[data-collapsed-port="${side}"]`)!;
      expect(Math.abs(centre(port.getBoundingClientRect()).y - letterBodyCentre(title))).toBeLessThanOrEqual(0.5);
    }
  });

  it('draws every edge, as itself, in an image export of collapsed nodes', async () => {
    const graph = graphWith(false);
    const exportRef = createRef<HTMLDivElement>();
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={createAdapter(createExecutionPort().port, projectSnapshotFor(graph))}>
            <Box css={flowThemeCss} h="full" w="full">
              <WorkflowImageExportView
                containerRef={exportRef}
                edges={toFlowEdges(graph, [], 'default', undefined, templates)}
                edgeTypes={{ default: WorkflowEdgeComponent }}
                nodes={toFlowNodes(graph, [], templates)}
                nodeTypes={nodeTypes}
              />
            </Box>
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );

    const exported = () =>
      [...exportRef.current!.querySelectorAll<SVGGElement>('.react-flow__edge')].map((element) => element.dataset.id);
    await vi.waitFor(() => expect(exported().sort()).toEqual(['ab1', 'ab2', 'ac']));
    expect(exportRef.current!.querySelector('[data-collapsed-edge-count]')).toBeNull();
    const strokes = [...exportRef.current!.querySelectorAll<SVGPathElement>('.react-flow__edge-path')].map(
      (path) => path.style.stroke
    );
    // IntegerField's colour, not the neutral a merged edge draws in.
    expect(new Set(strokes)).toEqual(new Set(['rgb(248, 113, 113)']));
  });

  it('keeps a collapsed node in a large graph’s compact mode unconnectable and cheap', async () => {
    const graph = graphWith(false);
    const execution = createExecutionPort('a');
    execution.set(completed(''));
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={createAdapter(execution.port, projectSnapshotFor(graph))}>
            <Box css={flowThemeCss} h="full" w="full">
              <ReactFlow
                edges={toFlowEdges(graph, [], 'default', undefined, templates)}
                edgeTypes={{ default: WorkflowEdgeComponent }}
                nodes={toFlowNodes(graph, [], templates, undefined, true)}
                nodeTypes={nodeTypes}
              />
            </Box>
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );

    expect(node('a').querySelector('[data-collapsed-port="right"]')).not.toBeNull();
    const handles = [...node('a').querySelectorAll<HTMLElement>('.react-flow__handle')];
    expect(handles.length).toBeGreaterThan(0);
    expect(handles.some((handle) => handle.classList.contains('connectable'))).toBe(false);
    // The completed mark is a plain image named for its state: no button, no tooltip to mount.
    const status = node('a').querySelector<HTMLElement>('[data-node-status-indicator="true"]')!;
    expect(status.getAttribute('role')).toBe('img');
    expect(status.getAttribute('aria-label')).toBe('Status: Completed');
    expect(status.closest('button, [data-scope="tooltip"]')).toBeNull();
  });
});

describe('CurrentImageFlowNode', () => {
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
    exportMocks.progressImage = null;
  });

  const image = (name: string) => ({
    height: 2,
    imageName: name,
    imageUrl: `${outputImage(2, 2)}#${name}`,
    queuedAt: '2026-01-01T00:00:00Z',
    sourceQueueItemId: 'queue-item',
    thumbnailUrl: outputImage(1, 1),
    width: 2,
  });
  const recent = image('recent.png');
  const chosen = image('chosen.png');
  /** A project port whose gallery values the test changes, as the Gallery widget does. */
  const createGalleryProject = (galleryValues: Record<string, unknown>) => {
    let snapshot = projectSnapshotFor(currentImageGraph, galleryValues);
    const listeners = new Set<() => void>();

    return {
      port: {
        getSnapshot: () => snapshot,
        subscribe: (listener: () => void) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      setGalleryValues: (next: Record<string, unknown>) => {
        snapshot = projectSnapshotFor(currentImageGraph, next);
        listeners.forEach((listener) => listener());
      },
    };
  };
  const render = (
    project: ReturnType<typeof createGalleryProject>,
    onDragStart: (event: DragStartEvent) => void = () => {}
  ) =>
    act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider
            adapter={
              { ...createAdapter(createExecutionPort().port), project: project.port } as unknown as WorkflowUiAdapter
            }
          >
            {/* Stands in for the shell's drag context, where gallery items are dragged. */}
            <DndContext onDragStart={onDragStart}>
              <ReactFlow edges={[]} nodes={currentImageNodes} nodeTypes={nodeTypes} />
            </DndContext>
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );
  const shownImage = () => host.querySelector<HTMLImageElement>('.react-flow__node img')?.src ?? null;

  it('shows the gallery’s selected image, passes over a selected video, and falls back to the latest image', async () => {
    const project = createGalleryProject({ recentImages: [recent, chosen] });
    await render(project);
    // Nothing selected: the project's most recent image.
    await vi.waitFor(() => expect(shownImage()).toBe(recent.imageUrl));

    await act(() =>
      project.setGalleryValues({ recentImages: [recent, chosen], selectedImageName: 'image:chosen.png' })
    );
    await vi.waitFor(() => expect(shownImage()).toBe(chosen.imageUrl));

    await act(() => project.setGalleryValues({ recentImages: [recent, chosen], selectedImageName: 'video:clip.mp4' }));
    await vi.waitFor(() => expect(shownImage()).toBe(recent.imageUrl));
  });

  it('offers the shown image as a gallery drag source', async () => {
    const onDragStart = vi.fn();
    await render(createGalleryProject({ recentImages: [recent] }), onDragStart);
    await vi.waitFor(() => expect(shownImage()).toBe(recent.imageUrl));
    const frame = host.querySelector('.react-flow__node img')!.getBoundingClientRect();
    const from = { x: frame.left + frame.width / 2, y: frame.top + frame.height / 2 };

    await commands.mouseDragPath([from, { x: from.x + 30, y: from.y + 30 }], window.innerWidth);
    expect(onDragStart).toHaveBeenCalledOnce();
    expect(onDragStart.mock.calls[0]![0].active.data.current).toEqual({
      items: [{ kind: 'image', name: 'recent.png' }],
      kind: 'gallery-item',
    });

    const source = host.querySelector('.react-flow__node img')!.closest<HTMLElement>('.nodrag');
    expect(source).not.toBeNull();
    // A pointer drag source only: no tab stop or button role for the canvas's keys to fight over.
    expect(source!.getAttribute('tabindex')).toBeNull();
    expect(source!.getAttribute('role')).toBeNull();
  });

  it('does not render again for gallery changes that leave the shown image alone', async () => {
    const renders = vi.spyOn(exportMocks, 'progressImage', 'get');
    const project = createGalleryProject({ recentImages: [recent], selectedBoardId: 'a' });
    await render(project);
    await vi.waitFor(() => expect(shownImage()).toBe(recent.imageUrl));
    const settled = renders.mock.calls.length;

    await act(() => project.setGalleryValues({ recentImages: [recent], selectedBoardId: 'b', galleryPage: 3 }));
    expect(renders.mock.calls.length).toBe(settled);
    renders.mockRestore();
  });
});
