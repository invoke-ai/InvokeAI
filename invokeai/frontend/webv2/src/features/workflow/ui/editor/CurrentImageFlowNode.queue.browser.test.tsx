import type { QueueItem } from '@features/queue/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

/* eslint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop -- each render mounts a fresh flow on purpose */
import { ChakraProvider } from '@chakra-ui/react';
// Drive the real production queue stores and runtimes directly to exercise multi-slot transitions without mock drift.
import { activeProgressTargetStore } from '@features/queue/data/activeProgressTargetStore';
import { itemProgressStore } from '@features/queue/data/itemProgressStore';
import { progressImageStore } from '@features/queue/data/progressImageStore';
import { queueItemProgressStore } from '@features/queue/data/progressStore';
import { createQueueRealtimeRuntime } from '@features/queue/data/realtimeRuntime';
import { createQueueCoordinator } from '@features/queue/runtime/coordinator';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { buildCurrentImageNode, createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { ReactFlow } from '@xyflow/react';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { CurrentImageFlowNode } from './CurrentImageFlowNode';
import { toFlowNodes } from './flowAdapters';

import '@xyflow/react/dist/style.css';

const i18nMock = vi.hoisted(() => ({
  translate: ((key: string) => key) as (key: string, options?: Record<string, unknown>) => string,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: i18nMock.translate,
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
  },
});
i18nMock.translate = (key, options) => i18n.t(key, options as never) as unknown as string;

const currentImageNode = buildCurrentImageNode({ x: 20, y: 20 });
const currentImageGraph = { ...createProjectGraph('current-image-queue-test'), nodes: [currentImageNode] };
const currentImageNodes = toFlowNodes(currentImageGraph, []);
const nodeTypes = {
  current_image: CurrentImageFlowNode,
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

const makeQueueItem = (id: string, backendItemIds: number[]): QueueItem =>
  ({
    backendItemIds,
    id,
    snapshot: {
      destination: 'gallery',
      graph: { label: 'Workflow' },
      sourceId: 'workflow',
      submittedAt: '2026-11-25T00:00:00Z',
    },
    status: 'running',
  }) as unknown as QueueItem;

const createProjectAdapter = ({
  galleryValues = {},
  isWorkflowRunning = true,
  projectId = 'project-a',
  queueItems,
}: {
  galleryValues?: Record<string, unknown>;
  isWorkflowRunning?: boolean;
  projectId?: string;
  queueItems: readonly QueueItem[];
}): WorkflowUiAdapter => {
  const snapshot = {
    activeWorkflow: { document: currentImageGraph },
    activeWorkflowId: currentImageGraph.id,
    galleryValues,
    id: projectId,
    isWorkflowRunning,
    projectGraph: currentImageGraph,
    queueItems,
    workflowValues: {},
    workflows: [{ document: currentImageGraph }],
  };

  return {
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
    nodeExecution: {
      get: () => null,
      getOrigin: () => ({ projectId, workflowId: currentImageGraph.id }),
      subscribe: () => () => {},
      subscribeOrigin: () => () => {},
    },
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
  } as unknown as WorkflowUiAdapter;
};

const createRealtimeHarness = (itemIds: number[]) => {
  const handlers = new Map<string, Set<(payload: unknown) => void>>();
  const emit = (event: string, payload: unknown) => {
    for (const listener of handlers.get(event) ?? []) {
      listener(payload);
    }
  };
  const backend = {
    connect: vi.fn(),
    disconnect: vi.fn(),
    enqueueGenerate: vi.fn().mockResolvedValue({
      batchId: 'batch-test',
      enqueued: itemIds.length,
      itemIds,
      requested: itemIds.length,
    }),
    getItem: vi.fn(),
    getResultImageNames: vi.fn().mockResolvedValue([]),
    listItemsByOrigin: vi.fn().mockResolvedValue([]),
    on: (event: string, listener: (payload: unknown) => void) => {
      const set = handlers.get(event) ?? new Set();
      set.add(listener);
      handlers.set(event, set);
      return () => set.delete(listener);
    },
    onConnectionChange: () => () => {},
    readProgressPreviews: vi.fn().mockResolvedValue([]),
  };
  const coordinator = createQueueCoordinator(
    { onGalleryRefresh: vi.fn() },
    {
      backend: backend as never,
      modelLoads: { completed: vi.fn(), reset: vi.fn(), started: vi.fn() },
      nodeExecution: {
        clearAll: vi.fn(),
        completed: vi.fn(),
        failed: vi.fn(),
        progress: vi.fn(),
        setOrigin: vi.fn(),
        settleRunning: vi.fn(),
        started: vi.fn(),
      },
      sweepIntervalMs: 60_000,
    }
  );
  const realtime = createQueueRealtimeRuntime({
    applyStatusChange: vi.fn(),
    backend: backend as never,
    invalidate: vi.fn(),
    progress: itemProgressStore,
    refreshModelCache: vi.fn(),
  });
  return { coordinator, emit, realtime };
};

const resetQueueStores = (): void => {
  activeProgressTargetStore.clear();
  progressImageStore.clear();
  queueItemProgressStore.clearAll?.();
  itemProgressStore.clearAll();
};

describe('CurrentImageFlowNode with production queue stores', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    resetQueueStores();
    host = document.createElement('div');
    host.style.cssText = 'width: 480px; height: 520px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    resetQueueStores();
  });

  const renderNode = (adapter: WorkflowUiAdapter) =>
    act(() => {
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <ReactFlow edges={[]} nodes={currentImageNodes} nodeTypes={nodeTypes} />
          </WorkflowUiProvider>
        </ChakraProvider>
      );
    });

  it('stays stably on the newest running slot across alternating multi-GPU frames and foreign project updates', async () => {
    const slot1FrameA = { dataUrl: 'data:image/png;base64,czFh', height: 2, width: 2 };
    const slot1FrameB = { dataUrl: 'data:image/png;base64,czFi', height: 2, width: 2 };
    const slot2FrameA = { dataUrl: 'data:image/png;base64,czJh', height: 2, width: 2 };
    const slot2FrameB = { dataUrl: 'data:image/png;base64,czJi', height: 2, width: 2 };
    const foreignFrame = { dataUrl: 'data:image/png;base64,b3RoZXI=', height: 2, width: 2 };
    const adapter = createProjectAdapter({
      queueItems: [makeQueueItem('batch-a', [101, 102])],
    });

    await renderNode(adapter);

    // Slot 1 @ 20%, then Slot 2 @ 61% -> follows newest running slot (Slot 2).
    await act(() => {
      activeProgressTargetStore.markFramed({ itemIndex: 1, queueItemId: 'batch-a' });
      progressImageStore.set(slot1FrameA, { itemIndex: 1, queueItemId: 'batch-a' });
      itemProgressStore.set(101, { message: 'Denoising', percentage: 0.2 });

      activeProgressTargetStore.markFramed({ itemIndex: 2, queueItemId: 'batch-a' });
      progressImageStore.set(slot2FrameA, { itemIndex: 2, queueItemId: 'batch-a' });
      itemProgressStore.set(102, { message: 'Denoising', percentage: 0.61 });
      queueItemProgressStore.set('batch-a', {
        activeItemIndex: 2,
        completedItemCount: 0,
        message: 'Denoising',
        percentage: 0.61,
        totalItemCount: 2,
      });
    });

    expect(host.querySelector('img')?.getAttribute('src')).toBe(slot2FrameA.dataUrl);
    expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('61%');

    // Slot 1 emits a newer frame @ 22% -> must NOT flip back to Slot 1.
    await act(() => {
      progressImageStore.set(slot1FrameB, { itemIndex: 1, queueItemId: 'batch-a' });
      itemProgressStore.set(101, { message: 'Denoising', percentage: 0.22 });
      queueItemProgressStore.set('batch-a', {
        activeItemIndex: 1,
        completedItemCount: 0,
        message: 'Denoising',
        percentage: 0.22,
        totalItemCount: 2,
      });
    });

    expect(host.querySelector('img')?.getAttribute('src')).toBe(slot2FrameA.dataUrl);
    expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('61%');

    // Slot 2 advances to 63%, then another project publishes a frame -> stays stably on Slot 2 @ 63%.
    await act(() => {
      progressImageStore.set(slot2FrameB, { itemIndex: 2, queueItemId: 'batch-a' });
      itemProgressStore.set(102, { message: 'Denoising', percentage: 0.63 });
      queueItemProgressStore.set('batch-a', {
        activeItemIndex: 2,
        completedItemCount: 0,
        message: 'Denoising',
        percentage: 0.63,
        totalItemCount: 2,
      });

      activeProgressTargetStore.markFramed({ itemIndex: 1, queueItemId: 'run-b' });
      progressImageStore.set(foreignFrame, { itemIndex: 1, queueItemId: 'run-b' });
    });

    expect(host.querySelector('img')?.getAttribute('src')).toBe(slot2FrameB.dataUrl);
    expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('63%');
  });

  it('switches from a finished settling slot to a still-running sibling and keeps its per-slot percentage', async () => {
    const slot1Frame = { dataUrl: 'data:image/png;base64,cnVubmluZzE=', height: 2, width: 2 };
    const slot2Frame = { dataUrl: 'data:image/png;base64,ZmluaXNoZWQy', height: 2, width: 2 };
    const adapter = createProjectAdapter({
      queueItems: [makeQueueItem('batch-e', [501, 502])],
    });

    await renderNode(adapter);

    await act(() => {
      activeProgressTargetStore.markFramed({ itemIndex: 1, queueItemId: 'batch-e' });
      progressImageStore.set(slot1Frame, { itemIndex: 1, queueItemId: 'batch-e' });
      itemProgressStore.set(501, { message: 'Denoising', percentage: 0.5 });

      activeProgressTargetStore.markFramed({ itemIndex: 2, queueItemId: 'batch-e' });
      progressImageStore.set(slot2Frame, { itemIndex: 2, queueItemId: 'batch-e' });
      itemProgressStore.set(502, { message: 'Finishing', percentage: 0.95 });
    });

    // Slot 2 finishes and enters settling while coordinator clears activeItemIndex; Slot 1 keeps running at 73%.
    await act(() => {
      activeProgressTargetStore.settle({ itemIndex: 2, queueItemId: 'batch-e' });
      itemProgressStore.clear(502);
      itemProgressStore.set(501, { message: 'Denoising', percentage: 0.73 });
      queueItemProgressStore.set('batch-e', {
        activeItemIndex: undefined,
        completedItemCount: 1,
        message: '',
        percentage: null,
        totalItemCount: 2,
      });
    });

    const badge = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"]')!;
    expect(host.querySelector('img')?.getAttribute('src')).toBe(slot1Frame.dataUrl);
    expect(badge.textContent).toBe('73%');
    expect(badge.querySelector('.chakra-spinner')).toBeNull();
  });

  it('hides stale frame and percentage on requeue across coordinator and realtime runtime until fresh progress arrives', async () => {
    await import('@platform/ui/video-player/VideoPlayer');
    const selectedVideo = {
      boardId: 'none',
      category: 'general',
      createdAt: '2026-11-23T00:00:00Z',
      durationSeconds: 4,
      fps: 24,
      fullUrl: 'data:audio/wav;base64,UklGRiUAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQEAAACA',
      height: 576,
      isIntermediate: false,
      kind: 'video' as const,
      name: 'clip.mp4',
      starred: false,
      thumbnailUrl: 'data:image/png;base64,c2F2ZWQ=',
      width: 1024,
    };
    const attempt1Frame = { dataUrl: 'data:image/png;base64,YXR0ZW1wdDE=', height: 2, width: 2 };
    const attempt2Frame = { dataUrl: 'data:image/png;base64,YXR0ZW1wdDI=', height: 2, width: 2 };
    const { coordinator, emit, realtime } = createRealtimeHarness([601]);

    try {
      coordinator.connect();
      realtime.start();
      await coordinator.submitGenerate('run-requeue', {
        batchCount: 1,
        destination: 'gallery',
        graph: { edges: [], id: 'g', nodes: {} },
        origin: 'webv2:p:project-a:q:run-requeue',
      } as never);

      await renderNode(
        createProjectAdapter({
          galleryValues: { selectedImage: selectedVideo },
          queueItems: [makeQueueItem('run-requeue', [601])],
        })
      );

      // Attempt 1 reaches 88% with a live frame.
      await act(() => {
        emit('invocation_progress', {
          image: { dataURL: attempt1Frame.dataUrl, height: 2, width: 2 },
          invocation_source_id: 'denoise',
          item_id: 601,
          message: 'Denoising',
          percentage: 0.88,
          revision: 1,
          session_id: 'session-1',
        });
      });
      expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('88%');
      expect(host.querySelector('img')?.getAttribute('src')).toBe(attempt1Frame.dataUrl);

      // Requeue returns to Waiting and restores the selected video.
      await act(() => {
        emit('queue_item_status_changed', { item_id: 601, status: 'pending', status_sequence: 1 });
      });
      await vi.waitFor(() => expect(host.querySelector('video')).not.toBeNull(), { timeout: 10_000 });
      expect(host.querySelector('[data-node-status-indicator="true"]')?.textContent).toBe(
        i18n.t('common.status.waiting')
      );

      // Attempt 2 restarts before any new progress/frame: keeps video visible and shows spinner (NO 88%, NO old frame).
      await act(() => {
        emit('queue_item_status_changed', { item_id: 601, status: 'in_progress', status_sequence: 2 });
      });
      const restartingBadge = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"]')!;
      expect(restartingBadge.textContent).toBe('');
      expect(restartingBadge.querySelector('.chakra-spinner')).not.toBeNull();
      expect(host.querySelector('video')).not.toBeNull();
      expect(host.querySelector(`img[src="${attempt1Frame.dataUrl}"]`)).toBeNull();

      // Fresh Attempt 2 progress + frame replaces the video with the new live preview at 15%.
      await act(() => {
        emit('invocation_progress', {
          image: { dataURL: attempt2Frame.dataUrl, height: 2, width: 2 },
          invocation_source_id: 'denoise',
          item_id: 601,
          message: 'Denoising',
          percentage: 0.15,
          revision: 1,
          session_id: 'session-2',
        });
      });
      expect(host.querySelector('img')?.getAttribute('src')).toBe(attempt2Frame.dataUrl);
      expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('15%');
    } finally {
      realtime.dispose();
      coordinator.dispose();
    }
  });

  it('keeps showing a saved-workflow child percentage while the parent item waits and cancels resize drags on project switch', async () => {
    const childFrame = { dataUrl: 'data:image/png;base64,Y2hpbGQ=', height: 2, width: 2 };
    const { coordinator, emit, realtime } = createRealtimeHarness([701]);
    const editGraph = vi.fn();

    try {
      coordinator.connect();
      realtime.start();
      await coordinator.submitGenerate('run-call-wf', {
        batchCount: 1,
        destination: 'gallery',
        graph: { edges: [], id: 'g', nodes: { call: { id: 'call', type: 'call_saved_workflow' } } },
        origin: 'webv2:p:project-a:q:run-call-wf',
      } as never);

      const adapterA = createProjectAdapter({
        projectId: 'project-a',
        queueItems: [makeQueueItem('run-call-wf', [701])],
      });
      adapterA.commands.editGraph = editGraph;
      await renderNode(adapterA);

      await act(() => {
        emit('queue_item_status_changed', { item_id: 701, status: 'waiting', status_sequence: 1 });
        emit('invocation_progress', {
          image: { dataURL: childFrame.dataUrl, height: 2, width: 2 },
          invocation_source_id: 'denoise',
          item_id: 702,
          message: 'Denoising child',
          percentage: 0.45,
          revision: 1,
          root_item_id: 701,
          session_id: 'child-session-1',
          workflow_call_parent_source_id: 'call',
        });
      });

      expect(host.querySelector('img')?.getAttribute('src')).toBe(childFrame.dataUrl);
      expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('45%');

      // Start resize drag on Project A, switch to Project B before pointerup -> no write to Project B.
      const handle = host.querySelector<HTMLElement>('[data-testid="current-image-node-resize-handle"]')!;
      await act(() => {
        handle.dispatchEvent(
          new PointerEvent('pointerdown', { bubbles: true, button: 0, clientX: 100, clientY: 100, pointerId: 1 })
        );
      });

      const adapterB = createProjectAdapter({ projectId: 'project-b', queueItems: [] });
      adapterB.commands.editGraph = editGraph;
      await renderNode(adapterB);

      await act(() => {
        handle.dispatchEvent(
          new PointerEvent('pointermove', { bubbles: true, clientX: 250, clientY: 250, pointerId: 1 })
        );
        handle.dispatchEvent(
          new PointerEvent('pointerup', { bubbles: true, clientX: 250, clientY: 250, pointerId: 1 })
        );
      });

      expect(editGraph).not.toHaveBeenCalled();
    } finally {
      realtime.dispose();
      coordinator.dispose();
    }
  });
});
