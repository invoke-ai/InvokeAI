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

const createProjectAdapter = ({
  galleryValues = {},
  isWorkflowRunning = true,
  projectId = 'project-a',
  queueItems,
}: {
  galleryValues?: Record<string, unknown>;
  isWorkflowRunning?: boolean;
  projectId?: string;
  queueItems: ReadonlyArray<{ backendItemIds?: readonly number[]; id: string }>;
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

  it('keeps the currently displayed local batch slot when another project publishes a frame', async () => {
    const slot1Frame = { dataUrl: 'data:image/png;base64,c2xvdDE=', height: 2, width: 2 };
    const slot2Frame = { dataUrl: 'data:image/png;base64,c2xvdDI=', height: 2, width: 2 };
    const otherProjectFrame = { dataUrl: 'data:image/png;base64,b3RoZXI=', height: 2, width: 2 };
    const adapter = createProjectAdapter({
      queueItems: [{ backendItemIds: [101, 102], id: 'batch-a' }],
    });

    await renderNode(adapter);

    // Slot 1 produces a frame first, then slot 2 produces a frame, then slot 1 updates again at 40% (slot 2 is at 75%).
    await act(() => {
      activeProgressTargetStore.markFramed({ itemIndex: 1, queueItemId: 'batch-a' });
      progressImageStore.set(slot1Frame, { itemIndex: 1, queueItemId: 'batch-a' });
      itemProgressStore.set(101, { message: 'Denoising', percentage: 0.2 });

      activeProgressTargetStore.markFramed({ itemIndex: 2, queueItemId: 'batch-a' });
      progressImageStore.set(slot2Frame, { itemIndex: 2, queueItemId: 'batch-a' });
      itemProgressStore.set(102, { message: 'Denoising', percentage: 0.75 });

      progressImageStore.set(slot1Frame, { itemIndex: 1, queueItemId: 'batch-a' });
      itemProgressStore.set(101, { message: 'Denoising', percentage: 0.4 });
      queueItemProgressStore.set('batch-a', {
        activeItemIndex: 1,
        completedItemCount: 0,
        message: 'Denoising',
        percentage: 0.4,
        totalItemCount: 2,
      });
    });

    expect(host.querySelector('img')?.getAttribute('src')).toBe(slot1Frame.dataUrl);
    expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('40%');

    // Another project publishes a frame: project A must keep showing slot 1 at 40% instead of jumping to slot 2 at 75%.
    await act(() => {
      activeProgressTargetStore.markFramed({ itemIndex: 1, queueItemId: 'run-b' });
      progressImageStore.set(otherProjectFrame, { itemIndex: 1, queueItemId: 'run-b' });
      queueItemProgressStore.set('run-b', {
        activeItemIndex: 1,
        completedItemCount: 0,
        message: 'Project B',
        percentage: 0.9,
        totalItemCount: 1,
      });
    });

    expect(host.querySelector('img')?.getAttribute('src')).toBe(slot1Frame.dataUrl);
    expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('40%');
  });

  it('hides the previous attempt retained frame when a requeued slot restarts before emitting a new frame', async () => {
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
    const previousAttemptFrame = { dataUrl: 'data:image/png;base64,b2xkQXR0ZW1wdA==', height: 2, width: 2 };
    const newAttemptFrame = { dataUrl: 'data:image/png;base64,bmV3QXR0ZW1wdA==', height: 2, width: 2 };
    const adapter = createProjectAdapter({
      galleryValues: { selectedImage: selectedVideo },
      queueItems: [{ backendItemIds: [201], id: 'run-a' }],
    });

    await renderNode(adapter);

    // Attempt 1 emits a frame.
    await act(() => {
      activeProgressTargetStore.markFramed({ itemIndex: 1, queueItemId: 'run-a' });
      progressImageStore.set(previousAttemptFrame, { itemIndex: 1, queueItemId: 'run-a' });
      queueItemProgressStore.set('run-a', {
        activeItemIndex: 1,
        completedItemCount: 0,
        message: 'Attempt 1',
        percentage: 0.6,
        totalItemCount: 1,
      });
    });
    expect(host.querySelector('img')?.getAttribute('src')).toBe(previousAttemptFrame.dataUrl);
    expect(host.querySelector('video')).toBeNull();

    // Requeue: coordinator clears activeProgressTargetStore (including framed flag) while progressImageStore retains the old frame.
    await act(() => {
      activeProgressTargetStore.clear({ itemIndex: 1, queueItemId: 'run-a' });
    });
    await vi.waitFor(
      async () => {
        await act(() => Promise.resolve());
        expect(host.querySelector('video')).not.toBeNull();
      },
      { timeout: 10_000 }
    );
    expect(host.querySelector('[data-node-status-indicator="true"]')?.textContent).toBe(
      i18n.t('common.status.waiting')
    );

    // Attempt 2 starts running (set only, not yet framed): selected video must stay visible instead of replaying Attempt 1's frame.
    await act(() => {
      activeProgressTargetStore.set({ itemIndex: 1, queueItemId: 'run-a' });
      queueItemProgressStore.set('run-a', {
        activeItemIndex: 1,
        completedItemCount: 0,
        message: 'Loading model',
        percentage: null,
        totalItemCount: 1,
      });
    });
    expect(host.querySelector('video')).not.toBeNull();
    expect(host.querySelector(`img[src="${previousAttemptFrame.dataUrl}"]`)).toBeNull();

    // Once Attempt 2 emits its first frame (markFramed + set), the live preview switches to Attempt 2's frame.
    await act(() => {
      progressImageStore.set(newAttemptFrame, { itemIndex: 1, queueItemId: 'run-a' });
      activeProgressTargetStore.markFramed({ itemIndex: 1, queueItemId: 'run-a' });
      queueItemProgressStore.set('run-a', {
        activeItemIndex: 1,
        completedItemCount: 0,
        message: 'Denoising',
        percentage: 0.25,
        totalItemCount: 1,
      });
    });
    expect(host.querySelector('video')).toBeNull();
    expect(host.querySelector('img')?.getAttribute('src')).toBe(newAttemptFrame.dataUrl);
    expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('25%');
  });

  it('keeps the remaining batch slot percentage when another batch slot finishes and clears coordinator activeItemIndex', async () => {
    const slot1Frame = { dataUrl: 'data:image/png;base64,Zmlyc3Q=', height: 2, width: 2 };
    const slot2Frame = { dataUrl: 'data:image/png;base64,c2Vjb25k', height: 2, width: 2 };
    const adapter = createProjectAdapter({
      queueItems: [{ backendItemIds: [301, 302], id: 'batch-c' }],
    });

    await renderNode(adapter);

    // Both batch slots run concurrently; slot 2 reaches 75%, then slot 1 finishes and clears the coordinator's active index & percentage.
    await act(() => {
      activeProgressTargetStore.markFramed({ itemIndex: 1, queueItemId: 'batch-c' });
      progressImageStore.set(slot1Frame, { itemIndex: 1, queueItemId: 'batch-c' });
      itemProgressStore.set(301, { message: 'Finishing', percentage: 0.95 });

      activeProgressTargetStore.markFramed({ itemIndex: 2, queueItemId: 'batch-c' });
      progressImageStore.set(slot2Frame, { itemIndex: 2, queueItemId: 'batch-c' });
      itemProgressStore.set(302, { message: 'Denoising', percentage: 0.75 });

      queueItemProgressStore.set('batch-c', {
        activeItemIndex: 1,
        completedItemCount: 0,
        message: 'Finishing',
        percentage: 0.95,
        totalItemCount: 2,
      });
    });

    // Slot 1 completes: coordinator clears slot 1 from activeProgressTargetStore/progressImageStore and publishes undefined activeItemIndex + null percentage.
    await act(() => {
      activeProgressTargetStore.clear({ itemIndex: 1, queueItemId: 'batch-c' });
      progressImageStore.clear({ itemIndex: 1, queueItemId: 'batch-c' });
      itemProgressStore.clear(301);
      queueItemProgressStore.set('batch-c', {
        activeItemIndex: undefined,
        completedItemCount: 1,
        message: '',
        percentage: null,
        totalItemCount: 2,
      });
    });

    const badge = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"]')!;
    expect(host.querySelector('img')?.getAttribute('src')).toBe(slot2Frame.dataUrl);
    expect(badge.textContent).toBe('75%');
    expect(badge.querySelector('.chakra-spinner')).toBeNull();
  });

  it('switches to a live framed sibling slot when a requeued slot restarts with a stale global frame and cancels resize drags on project switch', async () => {
    const staleSlot1Frame = { dataUrl: 'data:image/png;base64,c3RhbGUx', height: 2, width: 2 };
    const liveSlot2Frame = { dataUrl: 'data:image/png;base64,bGl2ZTI=', height: 2, width: 2 };
    const editGraph = vi.fn();
    const adapterA = createProjectAdapter({
      projectId: 'project-a',
      queueItems: [{ backendItemIds: [401, 402], id: 'batch-d' }],
    });
    adapterA.commands.editGraph = editGraph;

    await renderNode(adapterA);

    await act(() => {
      activeProgressTargetStore.markFramed({ itemIndex: 2, queueItemId: 'batch-d' });
      progressImageStore.set(liveSlot2Frame, { itemIndex: 2, queueItemId: 'batch-d' });
      itemProgressStore.set(402, { message: 'Denoising', percentage: 0.55 });

      // Slot 1 had an old global frame, was requeued (cleared), and restarted (set only, not yet framed).
      progressImageStore.set(staleSlot1Frame, { itemIndex: 1, queueItemId: 'batch-d' });
      activeProgressTargetStore.set({ itemIndex: 1, queueItemId: 'batch-d' });
    });

    expect(host.querySelector('img')?.getAttribute('src')).toBe(liveSlot2Frame.dataUrl);
    expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('55%');

    // Start a resize drag on project A, switch to project B before pointerup -> drag must not overwrite project B's node size.
    const handle = host.querySelector<HTMLElement>('[data-testid="current-image-node-resize-handle"]')!;
    await act(() => {
      handle.dispatchEvent(
        new PointerEvent('pointerdown', { bubbles: true, button: 0, clientX: 100, clientY: 100, pointerId: 1 })
      );
    });

    const adapterB = createProjectAdapter({
      projectId: 'project-b',
      queueItems: [],
    });
    adapterB.commands.editGraph = editGraph;
    await renderNode(adapterB);

    await act(() => {
      handle.dispatchEvent(
        new PointerEvent('pointermove', { bubbles: true, clientX: 250, clientY: 250, pointerId: 1 })
      );
      handle.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, clientX: 250, clientY: 250, pointerId: 1 }));
    });

    expect(editGraph).not.toHaveBeenCalled();
  });

  it('prefers a still-running framed batch slot over a finished settling slot that sent the latest frame', async () => {
    const finishedSlot1Frame = { dataUrl: 'data:image/png;base64,ZmluaXNoZWQx', height: 2, width: 2 };
    const runningSlot2Frame = { dataUrl: 'data:image/png;base64,cnVubmluZzI=', height: 2, width: 2 };
    const adapter = createProjectAdapter({
      queueItems: [{ backendItemIds: [501, 502], id: 'batch-e' }],
    });

    await renderNode(adapter);

    // Both slots emit frames; slot 1 sends the most recent global frame.
    await act(() => {
      activeProgressTargetStore.markFramed({ itemIndex: 2, queueItemId: 'batch-e' });
      progressImageStore.set(runningSlot2Frame, { itemIndex: 2, queueItemId: 'batch-e' });
      itemProgressStore.set(502, { message: 'Denoising', percentage: 0.5 });

      activeProgressTargetStore.markFramed({ itemIndex: 1, queueItemId: 'batch-e' });
      progressImageStore.set(finishedSlot1Frame, { itemIndex: 1, queueItemId: 'batch-e' });
      itemProgressStore.set(501, { message: 'Finishing', percentage: 0.95 });
      queueItemProgressStore.set('batch-e', {
        activeItemIndex: 1,
        completedItemCount: 0,
        message: 'Finishing',
        percentage: 0.95,
        totalItemCount: 2,
      });
    });

    // Slot 1 finishes and enters settling while its result loads; slot 2 keeps running at 73% without sending a new frame.
    await act(() => {
      activeProgressTargetStore.settle({ itemIndex: 1, queueItemId: 'batch-e' });
      itemProgressStore.clear(501);
      itemProgressStore.set(502, { message: 'Denoising', percentage: 0.73 });
      queueItemProgressStore.set('batch-e', {
        activeItemIndex: undefined,
        completedItemCount: 1,
        message: '',
        percentage: null,
        totalItemCount: 2,
      });
    });

    const badge = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"]')!;
    expect(host.querySelector('img')?.getAttribute('src')).toBe(runningSlot2Frame.dataUrl);
    expect(badge.textContent).toBe('73%');
    expect(badge.querySelector('.chakra-spinner')).toBeNull();
  });

  it('clears a requeued slot percentage across coordinator and realtime runtime so restart shows spinner until fresh progress', async () => {
    const attempt1Frame = { dataUrl: 'data:image/png;base64,YXR0ZW1wdDE=', height: 2, width: 2 };
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
        batchId: 'batch-requeue',
        enqueued: 1,
        itemIds: [601],
        requested: 1,
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

    try {
      coordinator.connect();
      realtime.start();
      await coordinator.submitGenerate('run-requeue', {
        batchCount: 1,
        destination: 'gallery',
        graph: { edges: [], id: 'g', nodes: {} },
        origin: 'webv2:p:project-a:q:run-requeue',
      } as never);

      const adapter = createProjectAdapter({
        queueItems: [{ backendItemIds: [601], id: 'run-requeue' }],
      });
      await renderNode(adapter);

      // Attempt 1 reaches 88% with a preview frame.
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

      // Requeue back to pending: both coordinator and itemProgressStore drop the 88%.
      await act(() => {
        emit('queue_item_status_changed', {
          item_id: 601,
          status: 'pending',
          status_sequence: 1,
        });
      });
      expect(host.querySelector('[data-node-status-indicator="true"]')?.textContent).toBe(
        i18n.t('common.status.waiting')
      );

      // Attempt 2 starts running before reporting any progress: must show spinner, NOT stale 88% or old frame.
      await act(() => {
        emit('queue_item_status_changed', {
          item_id: 601,
          status: 'in_progress',
          status_sequence: 2,
        });
      });
      const restartingBadge = host.querySelector<HTMLElement>('[data-testid="current-image-progress-badge"]')!;
      expect(restartingBadge.textContent).toBe('');
      expect(restartingBadge.querySelector('.chakra-spinner')).not.toBeNull();
      expect(host.querySelector(`img[src="${attempt1Frame.dataUrl}"]`)).toBeNull();

      // Fresh progress arrives on attempt 2 -> displays the new 15%.
      await act(() => {
        emit('invocation_progress', {
          invocation_source_id: 'denoise',
          item_id: 601,
          message: 'Denoising',
          percentage: 0.15,
          revision: 1,
          session_id: 'session-2',
        });
      });
      expect(host.querySelector('[data-testid="current-image-progress-badge"]')?.textContent).toBe('15%');
    } finally {
      realtime.dispose();
      coordinator.dispose();
    }
  });

  it('keeps showing a saved-workflow child percentage when the parent item enters waiting', async () => {
    const childFrame = { dataUrl: 'data:image/png;base64,Y2hpbGQ=', height: 2, width: 2 };
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
        batchId: 'batch-call-wf',
        enqueued: 1,
        itemIds: [701],
        requested: 1,
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

    try {
      coordinator.connect();
      realtime.start();
      await coordinator.submitGenerate('run-call-wf', {
        batchCount: 1,
        destination: 'gallery',
        graph: { edges: [], id: 'g', nodes: { call: { id: 'call', type: 'call_saved_workflow' } } },
        origin: 'webv2:p:project-a:q:run-call-wf',
      } as never);

      const adapter = createProjectAdapter({
        queueItems: [{ backendItemIds: [701], id: 'run-call-wf' }],
      });
      await renderNode(adapter);

      // Parent enters waiting while its child runs; realtime clears parent's itemProgressStore,
      // while coordinator keeps tracking the child's percentage and frame under root_item_id 701.
      await act(() => {
        emit('queue_item_status_changed', {
          item_id: 701,
          status: 'waiting',
          status_sequence: 1,
        });
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
    } finally {
      realtime.dispose();
      coordinator.dispose();
    }
  });
});
