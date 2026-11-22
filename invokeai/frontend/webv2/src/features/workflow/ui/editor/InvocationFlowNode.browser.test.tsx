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
import { GalleryUiProvider, type GalleryUiAdapter } from '@features/gallery/react';
import { galleryImageUrls } from '@features/gallery/utility';
import { parseOpenApiToTemplates } from '@features/workflow/data/templates';
import { flowThemeCss } from '@features/workflow/ui/graph-preview/flowTheme';
import { WorkflowImageExportProvider } from '@features/workflow/ui/nodeChrome';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import {
  buildInvocationNode,
  buildCurrentImageNode,
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
import { NotesFlowNode } from './NotesFlowNode';
import { WorkflowEdge as WorkflowEdgeComponent } from './WorkflowEdge';
import { exportWorkflowAsPng, WORKFLOW_EXPORT_TIMEOUT_MS } from './workflowImageExport';
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

  it('omits editor checkboxes, field validation, and failure diagnostics from the static export', async () => {
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
    const requiredTemplate: InvocationTemplate = {
      ...template,
      inputs: { a: { ...template.inputs.a!, input: 'direct', required: true } },
    };
    const imageNode = { ...documentNode, data: { ...documentNode.data, type: 'txt2img' }, id: 'image-node' };
    const incompleteNode: WorkflowInvocationNode = {
      ...documentNode,
      data: { ...documentNode.data, inputs: { a: { label: 'A', name: 'a', value: undefined } } },
      id: 'incomplete-node',
      position: { x: 340, y: 20 },
    };
    const failedCallNode = { ...callNode, position: { x: 20, y: 320 } };
    const graph = { ...projectGraph, nodes: [imageNode, incompleteNode, failedCallNode] };
    const nodes = toFlowNodes(graph, [], {
      call_saved_workflow: callTemplate,
      preview: requiredTemplate,
      txt2img: imageTemplate,
    });
    const execution = createExecutionPort(callNodeId);
    execution.set({
      error: 'Child node failed',
      latestOutput: null,
      outputImageName: null,
      progress: null,
      progressMessage: null,
      status: 'failed',
    });
    const adapter = createAdapter(execution.port, projectSnapshotFor(graph));
    let capturedClone: HTMLElement | undefined;
    exportMocks.rasterize.mockImplementation((clone: HTMLElement) => {
      capturedClone = clone;
      return Promise.resolve(new Blob(['png'], { type: 'image/png' }));
    });

    await render(adapter, 1, true, nodes);
    await exportWorkflowAsPng({
      bounds: { x: 20, y: 20, width: 620, height: 560 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host.querySelector<HTMLElement>('.react-flow')!,
      workflowName: 'Static export',
    });

    for (const id of ['image-node', 'incomplete-node', callNodeId]) {
      expect(capturedClone?.querySelector(`.react-flow__node[data-id="${id}"]`)).not.toBeNull();
    }
    expect(capturedClone?.textContent).not.toContain(i18n.t('nodes.useCache'));
    expect(capturedClone?.textContent).not.toContain(i18n.t('nodes.saveToGallery'));
    expect(capturedClone?.querySelector('input[type="checkbox"]')).toBeNull();
    expect(capturedClone?.querySelector('[data-part="error-text"]')).toBeNull();
    expect(capturedClone?.querySelector('[aria-invalid="true"]')).toBeNull();
    expect(capturedClone?.textContent).not.toContain('Required value.');
    expect(capturedClone?.querySelector('[data-node-status-indicator="true"]')).toBeNull();
    expect(capturedClone?.textContent).not.toContain('Failed');
    expect(capturedClone?.textContent).not.toContain('Child node failed');
    expect(capturedClone?.textContent).not.toContain('Child workflow error');
  });

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

  /** Mirrors WorkflowEditorView: the flow model rebuilds in a transition, so a commit echoes back a beat late. */
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
    expect(document.activeElement?.classList.contains('react-flow__node')).toBe(true);

    await rename('Discarded{Escape}');
    expect(fieldLabel()).toBe('Iterations');

    await rename('Steps{Enter}');
    expect(fieldLabel()).toBe('');
    expect(stepsTitle().textContent).toBe('Steps *');
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
    // xyflow keeps a node `visibility: hidden` until measured; hit-tests pass through it, so wait for every node.
    await vi.waitFor(() =>
      expect(
        [...host.querySelectorAll<HTMLElement>('.react-flow__node')].every(
          (element) => getComputedStyle(element).visibility === 'visible'
        )
      ).toBe(true)
    );
  };
  const node = (id: string) => host.querySelector<HTMLElement>(`.react-flow__node[data-id="${id}"]`)!;

  const statusButton = (id: string) => node(id).querySelector<HTMLElement>('[data-node-status-indicator="true"]');
  const panelText = () => document.querySelector('[data-scope="tooltip"][data-part="content"]')?.textContent ?? '';

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
    expect([...rowActions('B').querySelectorAll('button')].map((button) => button.getAttribute('aria-label'))).toEqual([
      'Edit field description',
    ]);
    expect(opacity(actions)).toBe('0');

    await act(() => userEvent.hover(fieldTitle('A')));
    await vi.waitFor(() => expect(opacity(actions)).toBe('1'));
    expect(opacity(rowActions('B'))).toBe('0');
    await act(() => userEvent.unhover(fieldTitle('A')));
    await vi.waitFor(() => expect(opacity(actions)).toBe('0'));

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

    const path = host.querySelector<SVGPathElement>('.react-flow__edge[data-id="ac"] .react-flow__edge-path')!;
    const start = path.getPointAtLength(0).matrixTransform(path.getScreenCTM()!);
    const port = sourcePort.getBoundingClientRect();
    expect(Math.abs(start.x - port.right)).toBeLessThan(1);
    expect(Math.abs(start.y - centre(port).y)).toBeLessThan(1);

    const hiddenHandle = node('a').querySelector<HTMLElement>('.react-flow__handle[data-handleid="value"]')!;
    expect(getComputedStyle(hiddenHandle).visibility).toBe('hidden');
    expect(hiddenHandle.classList.contains('connectable')).toBe(false);

    await render(graphWith(true));
    await vi.waitFor(() => expect(drawnEdgeIds().sort()).toEqual(['ab1', 'ab2', 'ac']));
    expect(host.querySelector('[data-collapsed-port]')).toBeNull();
    expect(host.querySelector('[data-collapsed-edge-count]')).toBeNull();
  });

  /** The middle of a label's lowercase letters: a 1ex inline box on the baseline spans exactly its x-height. */

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
    const status = node('a').querySelector<HTMLElement>('[data-node-status-indicator="true"]')!;
    expect(status.getAttribute('role')).toBe('img');
    expect(status.getAttribute('aria-label')).toBe('Status: Completed');
    expect(status.closest('button, [data-scope="tooltip"]')).toBeNull();
  });
});

describe('CurrentImageFlowNode', () => {
  let host: HTMLDivElement;
  let root: Root;
  const openItemContextMenu = vi.fn();
  const galleryAdapter = { openItemContextMenu } as unknown as GalleryUiAdapter;

  beforeEach(() => {
    openItemContextMenu.mockClear();
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
            <GalleryUiProvider adapter={galleryAdapter}>
              {/* Stands in for the shell's drag context, where gallery items are dragged. */}
              <DndContext onDragStart={onDragStart}>
                <ReactFlow edges={[]} nodes={currentImageNodes} nodeTypes={nodeTypes} />
              </DndContext>
            </GalleryUiProvider>
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );
  const shownImage = () => host.querySelector<HTMLImageElement>('.react-flow__node img')?.src ?? null;

  it('shows the gallery’s selected image, passes over a selected video, and falls back to the latest image', async () => {
    const project = createGalleryProject({ recentImages: [recent, chosen] });
    await render(project);
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
    expect(source!.getAttribute('tabindex')).toBeNull();
    expect(source!.getAttribute('role')).toBeNull();
  });

  it("opens the shown image's menu from a header button the keyboard reaches, only while an image is shown", async () => {
    const project = createGalleryProject({ recentImages: [recent] });
    const menuButton = () => host.querySelector<HTMLButtonElement>('button[aria-label="Image actions"]');
    await render(project);
    await vi.waitFor(() => expect(shownImage()).toBe(recent.imageUrl));

    const flowNode = host.querySelector<HTMLElement>('.react-flow__node')!;
    // The flow keeps a node hidden, so unfocusable, until it has measured it.
    await vi.waitFor(() => {
      flowNode.focus();
      expect(document.activeElement).toBe(flowNode);
    });
    await act(() => userEvent.keyboard('{Tab}'));
    expect(document.activeElement).toBe(menuButton());
    await act(() => userEvent.keyboard('{Enter}'));

    const button = menuButton()!.getBoundingClientRect();
    expect(openItemContextMenu).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ ref: { kind: 'image', name: 'recent.png' }, x: button.left, y: button.bottom })
    );
    expect(openItemContextMenu.mock.lastCall![0].returnFocus()).toBe(menuButton());

    // A pointer press opens it below the button too, not at the pointer.
    await act(() => userEvent.click(menuButton()!));
    expect(openItemContextMenu.mock.lastCall![0]).toMatchObject({ x: button.left, y: button.bottom });

    exportMocks.progressImage = { dataUrl: outputImage(1, 1), height: 1, width: 1 };
    await act(() => project.setGalleryValues({ recentImages: [chosen] }));
    expect(host.textContent).toContain('Generating');
    expect(menuButton()).toBeNull();

    exportMocks.progressImage = null;
    await act(() => project.setGalleryValues({}));
    expect(menuButton()).toBeNull();
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
