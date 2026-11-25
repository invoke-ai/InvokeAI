import type { QueueItem } from '@features/queue/contracts';
import type { ProjectGraphState } from '@features/workflow/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

/* eslint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop -- each render mounts a fresh flow on purpose */
import { ChakraProvider } from '@chakra-ui/react';
import { GalleryUiProvider, type GalleryUiAdapter } from '@features/gallery/react';
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

const testMocks = vi.hoisted(() => ({
  canCopyImage: vi.fn<() => boolean>(() => true),
  copyImage: vi.fn<(blob: Blob) => Promise<void>>(() => Promise.resolve()),
  refreshProtectedMediaCookie: vi.fn<() => Promise<boolean>>().mockResolvedValue(true),
  translate: ((key: string) => key) as (key: string, options?: Record<string, unknown>) => string,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: testMocks.translate,
  }),
}));

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getAuthSession: () => ({ accountEpoch: 1 }),
  refreshProtectedMediaCookie: testMocks.refreshProtectedMediaCookie,
}));

vi.mock('@platform/browser/clipboard', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  canCopyImage: () => testMocks.canCopyImage(),
  copyImage: (blob: Blob) => testMocks.copyImage(blob),
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
testMocks.translate = (key, options) => i18n.t(key, options as never) as unknown as string;

const currentImageNode = buildCurrentImageNode({ x: 20, y: 20 });
const currentImageGraph: ProjectGraphState = {
  ...createProjectGraph('current-image-queue-test'),
  nodes: [currentImageNode],
};
const nodeTypes = {
  current_image: CurrentImageFlowNode,
};

const SAMPLE_VIDEO = {
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
  editGraph = vi.fn(),
  galleryValues = {},
  graph = currentImageGraph,
  isWorkflowRunning = true,
  projectId = 'project-a',
  queueItems,
}: {
  editGraph?: ReturnType<typeof vi.fn>;
  galleryValues?: Record<string, unknown>;
  graph?: ProjectGraphState;
  isWorkflowRunning?: boolean;
  projectId?: string;
  queueItems: readonly QueueItem[];
}): WorkflowUiAdapter => {
  const snapshot = {
    activeWorkflow: { document: graph },
    activeWorkflowId: graph.id,
    galleryValues,
    id: projectId,
    isWorkflowRunning,
    projectGraph: graph,
    queueItems,
    workflowValues: {},
    workflows: [{ document: graph }],
  };

  return {
    capabilities: { getSnapshot: () => ({ canUseCache: true }), subscribe: () => () => {} },
    commands: {
      addWorkflow: vi.fn(),
      createWorkflow: vi.fn(),
      duplicateWorkflow: vi.fn(),
      editGraph,
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
      getOrigin: () => ({ projectId, workflowId: graph.id }),
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

const nextFrame = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      })
  );

describe('CurrentImageFlowNode with production queue stores', () => {
  let host: HTMLDivElement;
  let root: Root;
  const openItemContextMenu = vi.fn();
  const galleryAdapter = { openItemContextMenu } as unknown as GalleryUiAdapter;

  beforeEach(() => {
    resetQueueStores();
    openItemContextMenu.mockReset();
    testMocks.canCopyImage.mockReset().mockReturnValue(true);
    testMocks.copyImage.mockReset().mockResolvedValue(undefined);
    testMocks.refreshProtectedMediaCookie.mockReset().mockResolvedValue(true);
    delete document.documentElement.dataset.reduceMotion;
    host = document.createElement('div');
    host.style.cssText = 'width: 800px; height: 600px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    resetQueueStores();
    delete document.documentElement.dataset.reduceMotion;
  });

  const renderNode = (
    adapter: WorkflowUiAdapter,
    { graph = currentImageGraph, zoom = 1 }: { graph?: ProjectGraphState; zoom?: number } = {}
  ) =>
    act(() => {
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <GalleryUiProvider adapter={galleryAdapter}>
              <ReactFlow
                defaultViewport={{ x: 0, y: 0, zoom }}
                edges={[]}
                nodes={toFlowNodes(graph, [])}
                nodeTypes={nodeTypes}
              />
            </GalleryUiProvider>
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

  it('tracks the running sibling percentage when a framed slot is settling and the running sibling has no frame yet', async () => {
    const settlingFrame = { dataUrl: 'data:image/png;base64,c2V0dGxpbmc=', height: 2, width: 2 };
    const adapter = createProjectAdapter({
      queueItems: [makeQueueItem('batch-settle', [501, 502])],
    });

    await renderNode(adapter);

    // Slot 1 finished with a held frame (settling, stale 95% still in itemProgressStore); Slot 2 is running unframed at 24%.
    await act(() => {
      activeProgressTargetStore.markFramed({ itemIndex: 1, queueItemId: 'batch-settle' });
      progressImageStore.set(settlingFrame, { itemIndex: 1, queueItemId: 'batch-settle' });
      itemProgressStore.set(501, { message: 'Finishing', percentage: 0.95 });
      activeProgressTargetStore.settle({ itemIndex: 1, queueItemId: 'batch-settle' });

      activeProgressTargetStore.set({ itemIndex: 2, queueItemId: 'batch-settle' });
      itemProgressStore.set(502, { message: 'Preparing', percentage: 0.24 });
    });

    // Holds Slot 1's frame visually while reporting Slot 2's live 24% (never Slot 1's stale 95%).
    expect(host.querySelector('img')?.getAttribute('src')).toBe(settlingFrame.dataUrl);
    expect(host.querySelector('[data-node-status-indicator="true"]')?.textContent).toBe(
      i18n.t('widgets.workflow.generating')
    );
    expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('24%');

    // If Slot 2 has no percentage yet, show spinner rather than Slot 1's finished 95%.
    await act(() => {
      itemProgressStore.set(502, { message: 'Preparing', percentage: null });
    });
    const badge = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"]')!;
    expect(badge.textContent).toBe('');
    expect(badge.querySelector('.chakra-spinner')).not.toBeNull();
  });

  it('keeps percentages stable across parallel call_saved_workflow batch slots without flickering to spinner', async () => {
    const slot1Frame = { dataUrl: 'data:image/png;base64,Y2hpbGQx', height: 2, width: 2 };
    const slot2Frame = { dataUrl: 'data:image/png;base64,Y2hpbGQy', height: 2, width: 2 };
    const { coordinator, emit, realtime } = createRealtimeHarness([701, 702]);

    try {
      coordinator.connect();
      realtime.start();
      await coordinator.submitGenerate('batch-call-wf', {
        batchCount: 2,
        destination: 'gallery',
        graph: { edges: [], id: 'g', nodes: { call: { id: 'call', type: 'call_saved_workflow' } } },
        origin: 'webv2:p:project-a:q:batch-call-wf',
      } as never);

      await renderNode(
        createProjectAdapter({
          queueItems: [makeQueueItem('batch-call-wf', [701, 702])],
        })
      );

      // Both parent items transition to waiting (clearing their itemProgressStore entries) while children execute on 2 GPUs.
      await act(() => {
        emit('queue_item_status_changed', { item_id: 701, status: 'waiting', status_sequence: 1 });
        emit('queue_item_status_changed', { item_id: 702, status: 'waiting', status_sequence: 1 });

        emit('invocation_progress', {
          image: { dataURL: slot1Frame.dataUrl, height: 2, width: 2 },
          invocation_source_id: 'denoise',
          item_id: 801,
          message: 'Child 1',
          percentage: 0.4,
          revision: 1,
          root_item_id: 701,
          session_id: 'child-1',
          workflow_call_parent_source_id: 'call',
        });
        emit('invocation_progress', {
          image: { dataURL: slot2Frame.dataUrl, height: 2, width: 2 },
          invocation_source_id: 'denoise',
          item_id: 802,
          message: 'Child 2',
          percentage: 0.72,
          revision: 1,
          root_item_id: 702,
          session_id: 'child-2',
          workflow_call_parent_source_id: 'call',
        });
      });

      expect(host.querySelector('img')?.getAttribute('src')).toBe(slot2Frame.dataUrl);
      expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('72%');

      // Sibling Slot 1 child emits a progress tick (overwriting coordinator.activeItemIndex to 1):
      // Slot 2 is still displayed and MUST remain at 72% instead of flickering to a spinner.
      await act(() => {
        emit('invocation_progress', {
          invocation_source_id: 'denoise',
          item_id: 801,
          message: 'Child 1',
          percentage: 0.45,
          revision: 2,
          root_item_id: 701,
          session_id: 'child-1',
          workflow_call_parent_source_id: 'call',
        });
      });

      const badge = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"]')!;
      expect(host.querySelector('img')?.getAttribute('src')).toBe(slot2Frame.dataUrl);
      expect(badge.textContent).toBe('72%');
      expect(badge.querySelector('.chakra-spinner')).toBeNull();
    } finally {
      realtime.dispose();
      coordinator.dispose();
    }
  });

  it('hides stale frame and percentage on requeue across coordinator and realtime runtime until fresh progress arrives', async () => {
    await import('./CurrentImageVideoSurface');
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
          galleryValues: { selectedImage: SAMPLE_VIDEO },
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

      // Attempt 2 restarts before any new progress/frame: keeps video visible and shows spinner at top-left (not bottom-left).
      await act(() => {
        emit('queue_item_status_changed', { item_id: 601, status: 'in_progress', status_sequence: 2 });
      });
      const restartingBadge = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"]')!;
      expect(restartingBadge.textContent).toBe('');
      expect(restartingBadge.querySelector('.chakra-spinner')).not.toBeNull();
      expect(getComputedStyle(restartingBadge).top).toBe('8px');
      expect(host.querySelector('video')).not.toBeNull();
      expect(host.querySelector(`img[src="${attempt1Frame.dataUrl}"]`)).toBeNull();

      // Fresh Attempt 2 progress + frame replaces the video with the new live preview at 15% (badge moves back to bottom-left).
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
      const liveBadge = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"]')!;
      expect(liveBadge.textContent).toBe('15%');
      expect(getComputedStyle(liveBadge).bottom).toBe('8px');
    } finally {
      realtime.dispose();
      coordinator.dispose();
    }
  });

  it('resizes accurately at non-1x zoom via ResizeCorner, supports keyboard & reset, ignores click jitter, and aborts on project switch', async () => {
    const editGraph = vi.fn();
    const adapterA = createProjectAdapter({
      editGraph,
      isWorkflowRunning: false,
      projectId: 'project-a',
      queueItems: [],
    });

    await renderNode(adapterA, { zoom: 0.5 });

    const handle = host.querySelector<HTMLElement>('[data-testid="current-image-node-resize-handle"]')!;
    expect(handle.getAttribute('role')).toBe('separator');
    expect(handle.getAttribute('aria-valuemin')).toBe('200');
    expect(handle.getAttribute('aria-valuemax')).toBe('1600');
    expect(handle.getAttribute('aria-valuenow')).toBe('320');
    expect(handle.getAttribute('aria-valuetext')).toBe(
      i18n.t('widgets.floating.resizeValue', { height: 320, width: 320 })
    );

    // 1 px pointer jitter (below drag threshold) must NOT commit a 321x320 edit.
    await act(() => {
      handle.dispatchEvent(
        new PointerEvent('pointerdown', {
          bubbles: true,
          button: 0,
          buttons: 1,
          clientX: 100,
          clientY: 100,
          pointerId: 1,
        })
      );
    });
    await act(() => {
      window.dispatchEvent(
        new PointerEvent('pointermove', { bubbles: true, buttons: 1, clientX: 101, clientY: 100, pointerId: 1 })
      );
    });
    await nextFrame();
    await act(() => {
      window.dispatchEvent(
        new PointerEvent('pointerup', { bubbles: true, buttons: 0, clientX: 101, clientY: 100, pointerId: 1 })
      );
    });
    expect(editGraph).not.toHaveBeenCalled();

    // Positive pointer drag at 0.5x zoom: +60 screen px X -> +120 W (440), +40 screen px Y -> +80 H (400).
    await act(() => {
      handle.dispatchEvent(
        new PointerEvent('pointerdown', {
          bubbles: true,
          button: 0,
          buttons: 1,
          clientX: 100,
          clientY: 100,
          pointerId: 2,
        })
      );
    });
    expect(handle.hasAttribute('data-dragging')).toBe(true);
    await act(() => {
      window.dispatchEvent(
        new PointerEvent('pointermove', { bubbles: true, buttons: 1, clientX: 160, clientY: 140, pointerId: 2 })
      );
    });
    await nextFrame();
    await act(() => {
      window.dispatchEvent(
        new PointerEvent('pointerup', { bubbles: true, buttons: 0, clientX: 160, clientY: 140, pointerId: 2 })
      );
    });
    expect(handle.hasAttribute('data-dragging')).toBe(false);
    expect(editGraph).toHaveBeenCalledExactlyOnceWith({
      height: 400,
      nodeId: currentImageNode.id,
      type: 'setCurrentImageNodeSize',
      width: 440,
    });

    // Escape during drag cancels without committing.
    editGraph.mockClear();
    await act(() => {
      handle.dispatchEvent(
        new PointerEvent('pointerdown', {
          bubbles: true,
          button: 0,
          buttons: 1,
          clientX: 100,
          clientY: 100,
          pointerId: 3,
        })
      );
    });
    await act(() => {
      window.dispatchEvent(
        new PointerEvent('pointermove', { bubbles: true, buttons: 1, clientX: 180, clientY: 180, pointerId: 3 })
      );
    });
    await nextFrame();
    await act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Escape' }));
    });
    expect(handle.hasAttribute('data-dragging')).toBe(false);
    expect(editGraph).not.toHaveBeenCalled();

    // Keyboard resizing: ArrowRight (+32 px), Shift+ArrowDown (+64 px), Home (min), End (max).
    await act(() => {
      handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'ArrowRight' }));
    });
    expect(editGraph).toHaveBeenLastCalledWith({
      height: 320,
      nodeId: currentImageNode.id,
      type: 'setCurrentImageNodeSize',
      width: 352,
    });

    await act(() => {
      handle.dispatchEvent(
        new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'ArrowDown', shiftKey: true })
      );
    });
    expect(editGraph).toHaveBeenLastCalledWith({
      height: 384,
      nodeId: currentImageNode.id,
      type: 'setCurrentImageNodeSize',
      width: 320,
    });

    await act(() => {
      handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Home' }));
    });
    expect(editGraph).toHaveBeenLastCalledWith({
      height: 160,
      nodeId: currentImageNode.id,
      type: 'setCurrentImageNodeSize',
      width: 200,
    });

    await act(() => {
      handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'End' }));
    });
    expect(editGraph).toHaveBeenLastCalledWith({
      height: 1600,
      nodeId: currentImageNode.id,
      type: 'setCurrentImageNodeSize',
      width: 1600,
    });

    // Double-click resets size to default.
    await act(() => {
      handle.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true }));
    });
    expect(editGraph).toHaveBeenLastCalledWith({
      height: undefined,
      nodeId: currentImageNode.id,
      type: 'setCurrentImageNodeSize',
      width: undefined,
    });

    // Project switch mid-drag aborts the active drag and never commits to Project B.
    editGraph.mockClear();
    await act(() => {
      handle.dispatchEvent(
        new PointerEvent('pointerdown', {
          bubbles: true,
          button: 0,
          buttons: 1,
          clientX: 100,
          clientY: 100,
          pointerId: 4,
        })
      );
    });
    await act(() => {
      window.dispatchEvent(
        new PointerEvent('pointermove', { bubbles: true, buttons: 1, clientX: 200, clientY: 200, pointerId: 4 })
      );
    });
    await nextFrame();

    const adapterB = createProjectAdapter({
      editGraph,
      isWorkflowRunning: false,
      projectId: 'project-b',
      queueItems: [],
    });
    await renderNode(adapterB, { zoom: 0.5 });

    await act(() => {
      window.dispatchEvent(
        new PointerEvent('pointerup', { bubbles: true, buttons: 0, clientX: 200, clientY: 200, pointerId: 4 })
      );
    });
    expect(editGraph).not.toHaveBeenCalled();
  });

  it('keeps title visible, collapses reset badge, uses compact VideoPlayer, and preserves copy-unavailable tooltip at 200x160', async () => {
    await import('./CurrentImageVideoSurface');
    testMocks.canCopyImage.mockReturnValue(false);
    const editGraph = vi.fn();
    const minSizedGraph: ProjectGraphState = {
      ...currentImageGraph,
      nodes: [
        {
          ...currentImageNode,
          data: { ...currentImageNode.data, height: 160, width: 200 },
        },
      ],
    };
    const runItem = makeQueueItem('run-min', [911]);
    const adapter = createProjectAdapter({
      editGraph,
      galleryValues: { selectedImage: SAMPLE_VIDEO },
      graph: minSizedGraph,
      isWorkflowRunning: true,
      queueItems: [runItem],
    });

    await act(() => {
      activeProgressTargetStore.set({ itemIndex: 1, queueItemId: 'run-min' });
      queueItemProgressStore.set('run-min', {
        activeItemIndex: 1,
        completedItemCount: 0,
        message: 'Preparing',
        percentage: null,
        totalItemCount: 1,
      });
    });

    await renderNode(adapter, { graph: minSizedGraph });
    await vi.waitFor(() => expect(host.querySelector('video')).not.toBeNull(), { timeout: 10_000 });

    const shell = host.querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!;
    expect(Math.round(shell.getBoundingClientRect().width)).toBe(200);
    expect(Math.round(shell.getBoundingClientRect().height)).toBe(160);

    // Title keeps readable width even with Generating + Copy + Reset + Menu at 200 px width.
    const titleEl = host.querySelector<HTMLElement>('[data-testid="current-image-node-title"]')!;
    expect(titleEl).not.toBeNull();
    expect(titleEl.getBoundingClientRect().width).toBeGreaterThanOrEqual(24);

    // Reset button collapses to an icon-only header button with unambiguous wording and returns focus to ResizeCorner.
    const resetButton = host.querySelector<HTMLButtonElement>('[data-testid="current-image-node-reset-size"]')!;
    expect(resetButton.textContent).toBe('');
    expect(resetButton.getAttribute('aria-label')).toBe(
      i18n.t('widgets.workflow.currentImageResetSize', { height: 160, width: 200 })
    );
    const resizeHandle = host.querySelector<HTMLElement>('[data-testid="current-image-node-resize-handle"]')!;
    await act(() => {
      resetButton.click();
    });
    expect(editGraph).toHaveBeenCalledExactlyOnceWith({
      height: undefined,
      nodeId: currentImageNode.id,
      type: 'setCurrentImageNodeSize',
      width: undefined,
    });
    expect(document.activeElement).toBe(resizeHandle);

    // Copy button uses aria-disabled (not native disabled) so the non-secure-origin tooltip remains accessible.
    const copyButton = host.querySelector<HTMLButtonElement>('[data-testid="current-image-copy-video-frame"]')!;
    expect(copyButton.disabled).toBe(false);
    expect(copyButton.getAttribute('aria-disabled')).toBe('true');
    const describedBy = copyButton.getAttribute('aria-describedby');
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy!)?.textContent).toBe(i18n.t('common.copyUnavailable'));

    // VideoPlayer uses compact mode (no fullscreen control) at 200x160.
    expect(host.querySelector(`button[aria-label="${i18n.t('common.videoPlayer.fullscreen')}"]`)).toBeNull();
  });

  it('keeps waiting projects isolated from another project running and copies a ready video frame via header button', async () => {
    await import('./CurrentImageVideoSurface');
    const adapter = createProjectAdapter({
      galleryValues: { selectedImage: SAMPLE_VIDEO },
      isWorkflowRunning: true,
      queueItems: [makeQueueItem('this-project-waiting', [931])],
    });

    await act(() => {
      activeProgressTargetStore.set({ itemIndex: 1, queueItemId: 'other-project-run' });
      queueItemProgressStore.set('other-project-run', {
        activeItemIndex: 1,
        completedItemCount: 0,
        message: 'Other project',
        percentage: 0.55,
        totalItemCount: 1,
      });
    });

    await renderNode(adapter);
    await vi.waitFor(() => expect(host.querySelector('video')).not.toBeNull(), { timeout: 10_000 });

    expect(host.querySelector('[data-node-status-indicator="true"]')?.textContent).toBe(
      i18n.t('common.status.waiting')
    );
    expect(host.querySelector('[data-testid="current-image-media-drag-grip"]')).not.toBeNull();
    expect(host.querySelector('[data-testid="current-image-progress-badge"]')).toBeNull();

    // Before video frame is ready, copy button exposes copyCurrentFrameNotReady reason and blocks activation.
    const copyButton = host.querySelector<HTMLButtonElement>('[data-testid="current-image-copy-video-frame"]')!;
    expect(copyButton.getAttribute('aria-disabled')).toBe('true');

    // Simulate a ready video frame and click Copy current frame -> invokes copyImage and shows success toast.
    const video = host.querySelector('video')!;
    Object.defineProperty(video, 'readyState', { configurable: true, value: HTMLMediaElement.HAVE_CURRENT_DATA });
    Object.defineProperty(video, 'videoWidth', { configurable: true, value: 64 });
    Object.defineProperty(video, 'videoHeight', { configurable: true, value: 64 });
    const drawImageSpy = vi.spyOn(CanvasRenderingContext2D.prototype, 'drawImage').mockImplementation(() => {});
    try {
      await act(() => {
        video.dispatchEvent(new Event('loadeddata'));
      });
      expect(copyButton.getAttribute('aria-disabled')).toBeNull();
      await act(async () => {
        copyButton.click();
        await Promise.resolve();
      });
      await vi.waitFor(() => expect(testMocks.copyImage).toHaveBeenCalledOnce());
      expect(adapter.notifications.success).toHaveBeenCalledWith(i18n.t('widgets.preview.copyCurrentFrameSuccess'));
    } finally {
      drawImageSpy.mockRestore();
    }
  });

  it('bounds automatic video cookie refresh across partial loadeddata reloads, surfaces Retry, and honors Reduce Motion', async () => {
    await import('./CurrentImageVideoSurface');
    document.documentElement.dataset.reduceMotion = 'true';
    const runItem = makeQueueItem('run-motion', [922]);
    const adapter = createProjectAdapter({
      galleryValues: { selectedImage: SAMPLE_VIDEO },
      isWorkflowRunning: true,
      queueItems: [runItem],
    });

    await act(() => {
      activeProgressTargetStore.set({ itemIndex: 1, queueItemId: 'run-motion' });
      queueItemProgressStore.set('run-motion', {
        activeItemIndex: 1,
        completedItemCount: 0,
        message: 'Preparing',
        percentage: null,
        totalItemCount: 1,
      });
    });

    await renderNode(adapter);
    await vi.waitFor(() => expect(host.querySelector('video')).not.toBeNull(), { timeout: 10_000 });

    const spinner = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"] .chakra-spinner')!;
    expect(spinner).not.toBeNull();
    expect(getComputedStyle(spinner).animationDuration).toBe('2s');

    const video = host.querySelector('video')!;
    const loadSpy = vi.spyOn(video, 'load').mockImplementation(() => {});

    // First error triggers one automatic cookie refresh and reload.
    await act(() => {
      video.dispatchEvent(new Event('error'));
    });
    await vi.waitFor(() => {
      expect(testMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(1);
      expect(loadSpy).toHaveBeenCalledTimes(1);
    });
    expect(host.querySelector('[data-testid="current-image-video-failure"]')).toBeNull();

    // Partial load followed by a second error must NOT trigger another auto-refresh.
    await act(() => {
      video.dispatchEvent(new Event('loadeddata'));
      video.dispatchEvent(new Event('error'));
    });
    await vi.waitFor(() => {
      expect(host.querySelector('[data-testid="current-image-video-failure"]')).not.toBeNull();
    });
    expect(testMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(1);
    expect(loadSpy).toHaveBeenCalledTimes(1);

    // Manual Retry clears overlay, reloads, and rearms automatic refresh.
    const retryButton = host.querySelector<HTMLButtonElement>('[data-testid="current-image-video-failure"] button')!;
    await act(() => {
      retryButton.click();
    });
    expect(host.querySelector('[data-testid="current-image-video-failure"]')).toBeNull();
    expect(loadSpy).toHaveBeenCalledTimes(2);

    await act(() => {
      video.dispatchEvent(new Event('error'));
    });
    await vi.waitFor(() => {
      expect(testMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(2);
      expect(loadSpy).toHaveBeenCalledTimes(3);
    });
    loadSpy.mockRestore();
  });
});
