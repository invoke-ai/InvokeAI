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
const nodeTypes = { current_image: CurrentImageFlowNode };

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

const createRealtimeHarness = (itemIds: number[], onBackendItemComplete?: () => Promise<void>) => {
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
    { onBackendItemComplete, onGalleryRefresh: vi.fn() },
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

describe('CurrentImageFlowNode with production queue runtimes', () => {
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

  it('tracks parallel call_saved_workflow slots stably without flipping and resets percentage on next invocation start', async () => {
    const slot1Frame = 'data:image/png;base64,Y2hpbGQx';
    const slot2Frame = 'data:image/png;base64,Y2hpbGQy';
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

      await renderNode(createProjectAdapter({ queueItems: [makeQueueItem('batch-call-wf', [701, 702])] }));

      await act(() => {
        emit('queue_item_status_changed', { item_id: 701, status: 'waiting', status_sequence: 1 });
        emit('queue_item_status_changed', { item_id: 702, status: 'waiting', status_sequence: 1 });
        emit('invocation_progress', {
          image: { dataURL: slot1Frame, height: 2, width: 2 },
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
          image: { dataURL: slot2Frame, height: 2, width: 2 },
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

      expect(host.querySelector('img')?.getAttribute('src')).toBe(slot2Frame);
      expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('72%');

      // Sibling Slot 1 ticks (and a foreign project publishes a frame): Slot 2 stays stably displayed at 72%.
      await act(() => {
        emit('invocation_progress', {
          image: { dataURL: 'data:image/png;base64,Y2hpbGQxYg==', height: 2, width: 2 },
          invocation_source_id: 'denoise',
          item_id: 801,
          message: 'Child 1',
          percentage: 0.45,
          revision: 2,
          root_item_id: 701,
          session_id: 'child-1',
          workflow_call_parent_source_id: 'call',
        });
        activeProgressTargetStore.markFramed({ itemIndex: 1, queueItemId: 'foreign-run' });
        progressImageStore.set(
          { dataUrl: 'data:image/png;base64,Zm9yZWlnbg==', height: 2, width: 2 },
          { itemIndex: 1, queueItemId: 'foreign-run' }
        );
      });

      const badge = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"]')!;
      expect(host.querySelector('img')?.getAttribute('src')).toBe(slot2Frame);
      expect(badge.textContent).toBe('72%');
      expect(badge.querySelector('.chakra-spinner')).toBeNull();

      // Starting the next invocation (VAE decode) resets Slot 2 from percentage to spinner instead of sticking at 100%.
      await act(() => {
        emit('invocation_started', {
          invocation_source_id: 'vae_decode',
          item_id: 802,
          root_item_id: 702,
          session_id: 'child-2',
          workflow_call_parent_source_id: 'call',
        });
      });
      expect(badge.textContent).toBe('');
      expect(badge.querySelector('.chakra-spinner')).not.toBeNull();
    } finally {
      realtime.dispose();
      coordinator.dispose();
    }
  });

  it('hides stale frame and percentage on requeue and tracks a running sibling while an earlier slot settles', async () => {
    await import('./CurrentImageVideoSurface');
    let releaseRouting!: () => void;
    const routingPromise = new Promise<void>((resolve) => {
      releaseRouting = resolve;
    });
    const { coordinator, emit, realtime } = createRealtimeHarness([601, 602], () => routingPromise);
    const attempt1Frame = 'data:image/png;base64,YXR0ZW1wdDE=';
    const attempt2Frame = 'data:image/png;base64,YXR0ZW1wdDI=';

    try {
      coordinator.connect();
      realtime.start();
      await coordinator.submitGenerate('batch-transitions', {
        batchCount: 2,
        destination: 'gallery',
        graph: { edges: [], id: 'g', nodes: {} },
        origin: 'webv2:p:project-a:q:batch-transitions',
      } as never);

      await renderNode(
        createProjectAdapter({
          galleryValues: { selectedImage: SAMPLE_VIDEO },
          queueItems: [makeQueueItem('batch-transitions', [601, 602])],
        })
      );

      // Slot 1 reaches 88% with a frame (after video was ready to copy), then is requeued -> returns to Waiting with video visible and copy disabled again until fresh loadeddata.
      await vi.waitFor(() => expect(host.querySelector('video')).not.toBeNull(), { timeout: 10_000 });
      const initialVideo = host.querySelector('video')!;
      Object.defineProperty(initialVideo, 'readyState', {
        configurable: true,
        value: HTMLMediaElement.HAVE_CURRENT_DATA,
      });
      Object.defineProperty(initialVideo, 'videoWidth', { configurable: true, value: 64 });
      Object.defineProperty(initialVideo, 'videoHeight', { configurable: true, value: 64 });
      await act(() => initialVideo.dispatchEvent(new Event('loadeddata')));
      expect(
        host.querySelector('[data-testid="current-image-copy-video-frame"]')?.getAttribute('aria-disabled')
      ).toBeNull();

      await act(() => {
        emit('invocation_progress', {
          image: { dataURL: attempt1Frame, height: 2, width: 2 },
          invocation_source_id: 'denoise',
          item_id: 601,
          message: 'Denoising',
          percentage: 0.88,
          revision: 1,
          session_id: 'session-1',
        });
      });
      expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('88%');

      // Normal run (without call_saved_workflow) starting next invocation (VAE decode) resets 88% to spinner.
      await act(() => {
        emit('invocation_started', {
          invocation_source_id: 'vae_decode',
          item_id: 601,
          session_id: 'session-1',
        });
      });
      expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('');
      expect(host.querySelector('[data-testid="current-image-progress-badge"] .chakra-spinner')).not.toBeNull();

      await act(() => {
        emit('queue_item_status_changed', { item_id: 601, status: 'pending', status_sequence: 1 });
      });
      await vi.waitFor(() => expect(host.querySelector('video')).not.toBeNull(), { timeout: 10_000 });
      expect(host.querySelector('[data-node-status-indicator="true"]')?.textContent).toBe(
        i18n.t('common.status.waiting')
      );
      expect(host.querySelector('[data-testid="current-image-copy-video-frame"]')?.getAttribute('aria-disabled')).toBe(
        'true'
      );

      // A foreign project's active run must not flip this waiting project to Generating or hide its drag grip.
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
      expect(host.querySelector('[data-node-status-indicator="true"]')?.textContent).toBe(
        i18n.t('common.status.waiting')
      );
      expect(host.querySelector('[data-testid="current-image-media-drag-grip"]')).not.toBeNull();
      expect(host.querySelector('[data-testid="current-image-progress-badge"]')).toBeNull();

      // Slot 1 restarts unframed -> keeps video visible, places spinner badge top-left (clear of Play button).
      await act(() => {
        emit('queue_item_status_changed', { item_id: 601, status: 'in_progress', status_sequence: 2 });
      });
      const badge = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"]')!;
      expect(badge.textContent).toBe('');
      expect(badge.querySelector('.chakra-spinner')).not.toBeNull();
      expect(getComputedStyle(badge).top).toBe('8px');
      expect(host.querySelector(`img[src="${attempt1Frame}"]`)).toBeNull();

      // Slot 1 emits fresh frame at 95%, then completes (settling) while Slot 2 starts running unframed at 24%:
      // holds Slot 1's frame visually while reporting Slot 2's live 24% (never Slot 1's finished 95%).
      await act(() => {
        emit('invocation_progress', {
          image: { dataURL: attempt2Frame, height: 2, width: 2 },
          invocation_source_id: 'denoise',
          item_id: 601,
          message: 'Denoising',
          percentage: 0.95,
          revision: 1,
          session_id: 'session-2',
        });
        emit('queue_item_status_changed', { item_id: 601, status: 'completed', status_sequence: 3 });
        emit('invocation_progress', {
          invocation_source_id: 'denoise',
          item_id: 602,
          message: 'Denoising 2',
          percentage: 0.24,
          revision: 1,
          session_id: 'session-3',
        });
      });

      expect(host.querySelector('img')?.getAttribute('src')).toBe(attempt2Frame);
      expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('24%');
      expect(getComputedStyle(host.querySelector('[data-testid="current-image-progress-badge"]')!).bottom).toBe('8px');
    } finally {
      releaseRouting();
      realtime.dispose();
      coordinator.dispose();
    }
  });

  it('resizes at non-1x zoom via ResizeCorner without per-move re-renders, supports keyboard/reset, and targets the originating project', async () => {
    const editGraph = vi.fn();
    const adapterA = createProjectAdapter({
      editGraph,
      isWorkflowRunning: false,
      projectId: 'project-a',
      queueItems: [],
    });

    await renderNode(adapterA, { zoom: 0.5 });

    const shell = host.querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!;
    const handle = host.querySelector<HTMLElement>('[data-testid="current-image-node-resize-handle"]')!;
    expect(handle.getAttribute('role')).toBe('separator');
    expect(handle.getAttribute('aria-valuemin')).toBe('200');
    expect(handle.getAttribute('aria-valuemax')).toBe('1600');
    expect(handle.getAttribute('aria-valuenow')).toBe('320');
    expect(handle.getAttribute('aria-valuetext')).toBe(
      i18n.t('widgets.floating.resizeValue', { height: 320, width: 320 })
    );

    // 1 px pointer jitter (below threshold) commits nothing.
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
      window.dispatchEvent(
        new PointerEvent('pointermove', { bubbles: true, buttons: 1, clientX: 160, clientY: 140, pointerId: 2 })
      );
    });
    await nextFrame();
    expect(handle.hasAttribute('data-dragging')).toBe(true);
    expect(shell.style.getPropertyValue('--ci-preview-w')).toBe('440px');
    expect(shell.style.getPropertyValue('--ci-preview-h')).toBe('400px');

    await act(() => {
      window.dispatchEvent(
        new PointerEvent('pointerup', { bubbles: true, buttons: 0, clientX: 160, clientY: 140, pointerId: 2 })
      );
    });
    expect(handle.hasAttribute('data-dragging')).toBe(false);
    expect(editGraph).toHaveBeenCalledExactlyOnceWith(
      { height: 400, nodeId: currentImageNode.id, type: 'setCurrentImageNodeSize', width: 440 },
      { projectId: 'project-a', workflowId: currentImageGraph.id }
    );

    // Escape cancels without committing.
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
      window.dispatchEvent(
        new PointerEvent('pointermove', { bubbles: true, buttons: 1, clientX: 180, clientY: 180, pointerId: 3 })
      );
    });
    await nextFrame();
    await act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Escape' }));
    });
    expect(editGraph).not.toHaveBeenCalled();
    expect(shell.style.getPropertyValue('--ci-preview-w')).toBe('');

    // Keyboard steps & clamping: ArrowRight (+32), Shift+ArrowDown (+64), Home (min), End (max), double-click (reset).
    for (const [init, expected] of [
      [{ key: 'ArrowRight' }, { height: 320, width: 352 }],
      [
        { key: 'ArrowDown', shiftKey: true },
        { height: 384, width: 320 },
      ],
      [{ key: 'Home' }, { height: 160, width: 200 }],
      [{ key: 'End' }, { height: 1600, width: 1600 }],
    ] as const) {
      await act(() => {
        handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }));
      });
      expect(editGraph).toHaveBeenLastCalledWith(
        { nodeId: currentImageNode.id, type: 'setCurrentImageNodeSize', ...expected },
        { projectId: 'project-a', workflowId: currentImageGraph.id }
      );
    }

    await act(() => {
      handle.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true }));
    });
    expect(editGraph).toHaveBeenLastCalledWith(
      { height: undefined, nodeId: currentImageNode.id, type: 'setCurrentImageNodeSize', width: undefined },
      { projectId: 'project-a', workflowId: currentImageGraph.id }
    );

    // Switching projects mid-drag commits strictly against the originating project target, never Project B.
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
      window.dispatchEvent(
        new PointerEvent('pointermove', { bubbles: true, buttons: 1, clientX: 140, clientY: 140, pointerId: 4 })
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
        new PointerEvent('pointerup', { bubbles: true, buttons: 0, clientX: 140, clientY: 140, pointerId: 4 })
      );
    });
    expect(editGraph).toHaveBeenCalledExactlyOnceWith(
      { height: 400, nodeId: currentImageNode.id, type: 'setCurrentImageNodeSize', width: 400 },
      { projectId: 'project-a', workflowId: currentImageGraph.id }
    );
  });

  it('keeps 200x160 layout readable with compact VideoPlayer, accessible copy button, cookie recovery, and Reduce Motion', async () => {
    await import('./CurrentImageVideoSurface');
    document.documentElement.dataset.reduceMotion = 'true';
    testMocks.canCopyImage.mockReturnValue(false);
    const editGraph = vi.fn();
    const minSizedGraph: ProjectGraphState = {
      ...currentImageGraph,
      nodes: [{ ...currentImageNode, data: { ...currentImageNode.data, height: 160, width: 200 } }],
    };
    const { coordinator, emit, realtime } = createRealtimeHarness([911]);

    try {
      coordinator.connect();
      realtime.start();
      await coordinator.submitGenerate('run-min', {
        batchCount: 1,
        destination: 'gallery',
        graph: { edges: [], id: 'g', nodes: {} },
        origin: 'webv2:p:project-a:q:run-min',
      } as never);

      const adapter = createProjectAdapter({
        editGraph,
        galleryValues: { selectedImage: SAMPLE_VIDEO },
        graph: minSizedGraph,
        isWorkflowRunning: true,
        queueItems: [makeQueueItem('run-min', [911])],
      });

      await renderNode(adapter, { graph: minSizedGraph });
      await act(() => {
        emit('queue_item_status_changed', { item_id: 911, status: 'in_progress', status_sequence: 1 });
      });
      await vi.waitFor(() => expect(host.querySelector('video')).not.toBeNull(), { timeout: 10_000 });

      // At 200x160, "Generating" header text hides and reset collapses to an icon button, leaving ample title width.
      const shell = host.querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!;
      expect(Math.round(shell.getBoundingClientRect().width)).toBe(200);
      expect(Math.round(shell.getBoundingClientRect().height)).toBe(160);
      expect(host.querySelector('[data-node-status-indicator="true"]')).toBeNull();
      expect(
        host.querySelector<HTMLElement>('[data-testid="current-image-node-title"]')!.getBoundingClientRect().width
      ).toBeGreaterThanOrEqual(48);

      const spinner = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"] .chakra-spinner')!;
      expect(getComputedStyle(spinner).animationDuration).toBe('2s');

      const resetButton = host.querySelector<HTMLButtonElement>('[data-testid="current-image-node-reset-size"]')!;
      expect(resetButton.textContent).toBe('');
      await act(() => resetButton.click());
      expect(editGraph).toHaveBeenCalledExactlyOnceWith(
        { height: undefined, nodeId: currentImageNode.id, type: 'setCurrentImageNodeSize', width: undefined },
        { projectId: 'project-a', workflowId: minSizedGraph.id }
      );
      expect(document.activeElement).toBe(host.querySelector('[data-testid="current-image-node-resize-handle"]'));

      // Copy button uses aria-disabled (not native disabled) so the non-secure-origin tooltip remains accessible.
      const copyButton = host.querySelector<HTMLButtonElement>('[data-testid="current-image-copy-video-frame"]')!;
      expect(copyButton.disabled).toBe(false);
      expect(copyButton.getAttribute('aria-disabled')).toBe('true');
      expect(document.getElementById(copyButton.getAttribute('aria-describedby')!)?.textContent).toBe(
        i18n.t('common.copyUnavailable')
      );

      // Compact VideoPlayer omits Fullscreen; bounded cookie refresh triggers once before Retry overlay.
      expect(host.querySelector(`button[aria-label="${i18n.t('common.videoPlayer.fullscreen')}"]`)).toBeNull();
      const video = host.querySelector('video')!;
      const loadSpy = vi.spyOn(video, 'load').mockImplementation(() => {});
      try {
        await act(() => video.dispatchEvent(new Event('error')));
        await vi.waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
        await act(() => {
          video.dispatchEvent(new Event('loadeddata'));
          video.dispatchEvent(new Event('error'));
        });
        await vi.waitFor(() =>
          expect(host.querySelector('[data-testid="current-image-video-failure"]')).not.toBeNull()
        );
        expect(testMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(1);

        // Manual retry clears overlay and rearms auto-refresh.
        await act(() =>
          host.querySelector<HTMLButtonElement>('[data-testid="current-image-video-failure"] button')!.click()
        );
        expect(host.querySelector('[data-testid="current-image-video-failure"]')).toBeNull();
        expect(loadSpy).toHaveBeenCalledTimes(2);

        await act(() => video.dispatchEvent(new Event('error')));
        await vi.waitFor(() => {
          expect(testMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(2);
          expect(loadSpy).toHaveBeenCalledTimes(3);
        });
      } finally {
        loadSpy.mockRestore();
      }

      // With clipboard enabled but video not yet ready, copy button exposes copyCurrentFrameNotReady before loadeddata.
      testMocks.canCopyImage.mockReturnValue(true);
      await renderNode(adapter, { graph: minSizedGraph });
      expect(copyButton.getAttribute('aria-disabled')).toBe('true');
      expect(document.getElementById(copyButton.getAttribute('aria-describedby')!)?.textContent).toBe(
        i18n.t('widgets.preview.copyCurrentFrameNotReady')
      );

      Object.defineProperty(video, 'readyState', { configurable: true, value: HTMLMediaElement.HAVE_CURRENT_DATA });
      Object.defineProperty(video, 'videoWidth', { configurable: true, value: 64 });
      Object.defineProperty(video, 'videoHeight', { configurable: true, value: 64 });
      const drawImageSpy = vi.spyOn(CanvasRenderingContext2D.prototype, 'drawImage').mockImplementation(() => {});
      try {
        await act(() => video.dispatchEvent(new Event('loadeddata')));
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
    } finally {
      realtime.dispose();
      coordinator.dispose();
    }
  });
});
