import type { QueueItem } from '@features/queue/contracts';
import type { ProjectGraphState } from '@features/workflow/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

/* eslint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop -- each render mounts a fresh flow on purpose */
import { ChakraProvider } from '@chakra-ui/react';
import { GalleryUiProvider, type GalleryUiAdapter } from '@features/gallery/react';
import { activeProgressTargetStore } from '@features/queue/data/activeProgressTargetStore';
import { progressImageStore } from '@features/queue/data/progressImageStore';
import { queueItemProgressStore } from '@features/queue/data/progressStore';
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
  refreshProtectedMediaCookie: vi.fn<() => Promise<boolean>>().mockResolvedValue(false),
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

const createCoordinatorHarness = (itemIds: number[], onBackendItemComplete?: () => Promise<void>) => {
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
  return { coordinator, emit };
};

const resetQueueStores = (): void => {
  activeProgressTargetStore.clear();
  progressImageStore.clear();
  queueItemProgressStore.clearAll?.();
};

const nextFrame = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      })
  );

describe('CurrentImageFlowNode with production queue coordinator', () => {
  let host: HTMLDivElement;
  let root: Root;
  const openItemContextMenu = vi.fn();
  const galleryAdapter = { openItemContextMenu } as unknown as GalleryUiAdapter;

  beforeEach(() => {
    resetQueueStores();
    openItemContextMenu.mockReset();
    testMocks.canCopyImage.mockReset().mockReturnValue(true);
    testMocks.copyImage.mockReset().mockResolvedValue(undefined);
    testMocks.refreshProtectedMediaCookie.mockReset().mockResolvedValue(false);
    host = document.createElement('div');
    host.style.cssText = 'width: 800px; height: 600px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    resetQueueStores();
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

  it('tracks live frames and slot percentages across parallel slots, settling handoff, and requeue', async () => {
    await import('./CurrentImageVideoSurface');
    let releaseRouting!: () => void;
    const routingPromise = new Promise<void>((resolve) => {
      releaseRouting = resolve;
    });
    const { coordinator, emit } = createCoordinatorHarness([601, 602], () => routingPromise);
    const attempt1Frame = 'data:image/png;base64,YXR0ZW1wdDE=';
    const attempt2Frame = 'data:image/png;base64,YXR0ZW1wdDI=';

    try {
      coordinator.connect();
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

      await act(() => {
        activeProgressTargetStore.set({ itemIndex: 1, queueItemId: 'other-project-run' });
        queueItemProgressStore.set('other-project-run', {
          activeItemIndex: 1,
          completedItemCount: 0,
          message: 'Other project',
          percentage: 0.55,
          slotPercentages: { 1: 0.55 },
          totalItemCount: 1,
        });
      });
      expect(host.querySelector('[data-node-status-indicator="true"]')?.textContent).toBe(
        i18n.t('common.status.waiting')
      );
      expect(host.querySelector('[data-testid="current-image-progress-badge"]')).toBeNull();

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
    } finally {
      releaseRouting();
      coordinator.dispose();
    }
  });

  it('updates rendered size, size pill, and ARIA live during zoom-aware drag and holds preview until commit', async () => {
    const editGraph = vi.fn();
    const adapter = createProjectAdapter({
      editGraph,
      isWorkflowRunning: false,
      projectId: 'project-a',
      queueItems: [],
    });

    await renderNode(adapter, { zoom: 0.5 });

    const shell = host.querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!;
    const handle = host.querySelector<HTMLElement>('[data-testid="current-image-node-resize-handle"]')!;
    expect(handle.getAttribute('role')).toBe('separator');
    expect(handle.getAttribute('aria-valuemin')).toBe('200');
    expect(handle.getAttribute('aria-valuemax')).toBe('1600');
    expect(handle.getAttribute('aria-valuenow')).toBe('320');
    expect(host.querySelector('[data-testid="current-image-node-reset-size"]')).toBeNull();

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
        new PointerEvent('pointermove', { bubbles: true, buttons: 1, clientX: 160, clientY: 140, pointerId: 1 })
      );
    });
    await nextFrame();
    expect(handle.hasAttribute('data-dragging')).toBe(true);
    expect(getComputedStyle(shell).width).toBe('440px');
    expect(getComputedStyle(shell).height).toBe('400px');
    expect(Math.round(shell.getBoundingClientRect().width)).toBe(220);
    expect(Math.round(shell.getBoundingClientRect().height)).toBe(200);
    expect(host.querySelector('[data-testid="current-image-node-reset-size"]')?.textContent).toBe('440×400 ↺');
    expect(handle.getAttribute('aria-valuenow')).toBe('440');
    expect(handle.getAttribute('aria-valuetext')).toBe(
      i18n.t('widgets.floating.resizeValue', { height: 400, width: 440 })
    );

    await act(() => {
      window.dispatchEvent(
        new PointerEvent('pointerup', { bubbles: true, buttons: 0, clientX: 160, clientY: 140, pointerId: 1 })
      );
    });
    expect(handle.hasAttribute('data-dragging')).toBe(false);
    expect(editGraph).toHaveBeenCalledExactlyOnceWith(
      { height: 400, nodeId: currentImageNode.id, type: 'setCurrentImageNodeSize', width: 440 },
      { projectId: 'project-a', workflowId: currentImageGraph.id }
    );
    await nextFrame();
    await nextFrame();
    expect(getComputedStyle(shell).width).toBe('440px');
    expect(getComputedStyle(shell).height).toBe('400px');

    const committedGraph: ProjectGraphState = {
      ...currentImageGraph,
      nodes: [{ ...currentImageNode, data: { ...currentImageNode.data, height: 400, width: 440 } }],
    };
    await renderNode(
      createProjectAdapter({
        editGraph,
        graph: committedGraph,
        isWorkflowRunning: false,
        projectId: 'project-a',
        queueItems: [],
      }),
      { graph: committedGraph, zoom: 0.5 }
    );
    expect(getComputedStyle(shell).width).toBe('440px');
    expect(getComputedStyle(shell).height).toBe('400px');

    await renderNode(adapter, { graph: currentImageGraph, zoom: 0.5 });
    expect(getComputedStyle(shell).width).toBe('320px');
    expect(getComputedStyle(shell).height).toBe('320px');
  });

  it('releases the held post-drag preview after the fallback timer when a commit is rejected', async () => {
    const editGraph = vi.fn();
    const adapter = createProjectAdapter({
      editGraph,
      isWorkflowRunning: false,
      projectId: 'project-a',
      queueItems: [],
    });

    await renderNode(adapter, { zoom: 0.5 });

    const shell = host.querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!;
    const handle = host.querySelector<HTMLElement>('[data-testid="current-image-node-resize-handle"]')!;

    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    try {
      await act(() => {
        handle.dispatchEvent(
          new PointerEvent('pointerdown', {
            bubbles: true,
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 100,
            pointerId: 10,
          })
        );
        window.dispatchEvent(
          new PointerEvent('pointermove', { bubbles: true, buttons: 1, clientX: 150, clientY: 130, pointerId: 10 })
        );
      });
      await act(() => {
        window.dispatchEvent(
          new PointerEvent('pointerup', { bubbles: true, buttons: 0, clientX: 150, clientY: 130, pointerId: 10 })
        );
      });
      expect(getComputedStyle(shell).width).toBe('420px');
      await act(() => {
        vi.advanceTimersByTime(500);
      });
      expect(getComputedStyle(shell).width).toBe('320px');
    } finally {
      vi.useRealTimers();
    }
  });

  it('ignores sub-threshold pointer jitter, cancels on Escape, and aborts cleanly on unmount', async () => {
    const editGraph = vi.fn();
    const adapter = createProjectAdapter({
      editGraph,
      isWorkflowRunning: false,
      projectId: 'project-a',
      queueItems: [],
    });

    await renderNode(adapter, { zoom: 0.5 });

    const shell = host.querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!;
    const handle = host.querySelector<HTMLElement>('[data-testid="current-image-node-resize-handle"]')!;

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
    expect(getComputedStyle(shell).width).toBe('320px');

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
        new PointerEvent('pointermove', { bubbles: true, buttons: 1, clientX: 150, clientY: 150, pointerId: 4 })
      );
    });
    await nextFrame();
    await act(() => {
      root.render(null);
    });
    await act(() => {
      window.dispatchEvent(
        new PointerEvent('pointerup', { bubbles: true, buttons: 0, clientX: 150, clientY: 150, pointerId: 4 })
      );
    });
    expect(editGraph).not.toHaveBeenCalled();
  });

  it('supports keyboard resize steps, clamping, and double-click reset', async () => {
    const editGraph = vi.fn();
    const adapter = createProjectAdapter({
      editGraph,
      isWorkflowRunning: false,
      projectId: 'project-a',
      queueItems: [],
    });

    await renderNode(adapter, { zoom: 0.5 });

    const handle = host.querySelector<HTMLElement>('[data-testid="current-image-node-resize-handle"]')!;
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

    const minGraph: ProjectGraphState = {
      ...currentImageGraph,
      nodes: [{ ...currentImageNode, data: { ...currentImageNode.data, height: 160, width: 200 } }],
    };
    editGraph.mockClear();
    await renderNode(
      createProjectAdapter({
        editGraph,
        graph: minGraph,
        isWorkflowRunning: false,
        projectId: 'project-a',
        queueItems: [],
      }),
      { graph: minGraph, zoom: 0.5 }
    );
    const minShell = host.querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!;
    await act(() => {
      handle.dispatchEvent(
        new PointerEvent('pointerdown', {
          bubbles: true,
          button: 0,
          buttons: 1,
          clientX: 100,
          clientY: 100,
          pointerId: 7,
        })
      );
      window.dispatchEvent(
        new PointerEvent('pointermove', { bubbles: true, buttons: 1, clientX: 160, clientY: 140, pointerId: 7 })
      );
    });
    await nextFrame();
    expect(getComputedStyle(minShell).width).toBe('320px');

    await act(() => {
      handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Home' }));
    });
    expect(handle.hasAttribute('data-dragging')).toBe(false);
    expect(getComputedStyle(minShell).width).toBe('200px');
    expect(editGraph).not.toHaveBeenCalled();
  });

  it('hides status labels at compact width, collapses reset to an icon, and integrates video copy and failure UI', async () => {
    await import('./CurrentImageVideoSurface');
    testMocks.canCopyImage.mockReturnValue(false);
    const editGraph = vi.fn();
    const minSizedGraph: ProjectGraphState = {
      ...currentImageGraph,
      nodes: [{ ...currentImageNode, data: { ...currentImageNode.data, height: 160, width: 200 } }],
    };
    const { coordinator, emit } = createCoordinatorHarness([911]);

    try {
      coordinator.connect();
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
      await vi.waitFor(() => expect(host.querySelector('video')).not.toBeNull(), { timeout: 10_000 });

      const shell = host.querySelector<HTMLElement>('[data-workflow-node-shell="true"]')!;
      expect(Math.round(shell.getBoundingClientRect().width)).toBe(200);
      expect(Math.round(shell.getBoundingClientRect().height)).toBe(160);
      expect(host.querySelector('[data-node-status-indicator="true"]')).toBeNull();

      await act(() => {
        emit('queue_item_status_changed', { item_id: 911, status: 'in_progress', status_sequence: 1 });
      });
      expect(host.querySelector('[data-node-status-indicator="true"]')).toBeNull();
      expect(
        host.querySelector<HTMLElement>('[data-testid="current-image-node-title"]')!.getBoundingClientRect().width
      ).toBeGreaterThanOrEqual(48);

      const resetButton = host.querySelector<HTMLButtonElement>('[data-testid="current-image-node-reset-size"]')!;
      expect(resetButton.textContent).toBe('');
      await act(() => resetButton.click());
      expect(editGraph).toHaveBeenCalledExactlyOnceWith(
        { height: undefined, nodeId: currentImageNode.id, type: 'setCurrentImageNodeSize', width: undefined },
        { projectId: 'project-a', workflowId: minSizedGraph.id }
      );
      expect(document.activeElement).toBe(host.querySelector('[data-testid="current-image-node-resize-handle"]'));

      const copyButton = host.querySelector<HTMLButtonElement>('[data-testid="current-image-copy-video-frame"]')!;
      expect(copyButton.disabled).toBe(false);
      expect(copyButton.getAttribute('aria-disabled')).toBe('true');
      expect(document.getElementById(copyButton.getAttribute('aria-describedby')!)?.textContent).toBe(
        i18n.t('common.copyUnavailable')
      );

      const video = host.querySelector('video')!;
      await act(() => video.dispatchEvent(new Event('error')));
      await vi.waitFor(() => expect(host.querySelector('[data-testid="current-image-video-failure"]')).not.toBeNull());

      testMocks.canCopyImage.mockReturnValue(true);
      await act(() => root.render(null));
      await renderNode(adapter, { graph: minSizedGraph });
      await vi.waitFor(() => expect(host.querySelector('video')).not.toBeNull(), { timeout: 10_000 });
      const secureCopyButton = host.querySelector<HTMLButtonElement>('[data-testid="current-image-copy-video-frame"]')!;
      expect(secureCopyButton.getAttribute('aria-disabled')).toBe('true');
      expect(document.getElementById(secureCopyButton.getAttribute('aria-describedby')!)?.textContent).toBe(
        i18n.t('widgets.preview.copyCurrentFrameNotReady')
      );

      const secureVideo = host.querySelector('video')!;
      Object.defineProperty(secureVideo, 'readyState', {
        configurable: true,
        value: HTMLMediaElement.HAVE_CURRENT_DATA,
      });
      Object.defineProperty(secureVideo, 'videoWidth', { configurable: true, value: 64 });
      Object.defineProperty(secureVideo, 'videoHeight', { configurable: true, value: 64 });
      const drawImageSpy = vi.spyOn(CanvasRenderingContext2D.prototype, 'drawImage').mockImplementation(() => {});
      try {
        await act(() => secureVideo.dispatchEvent(new Event('loadeddata')));
        expect(secureCopyButton.getAttribute('aria-disabled')).toBeNull();
        await act(async () => {
          secureCopyButton.click();
          await Promise.resolve();
        });
        await vi.waitFor(() => expect(testMocks.copyImage).toHaveBeenCalledOnce());
        expect(adapter.notifications.success).toHaveBeenCalledWith(i18n.t('widgets.preview.copyCurrentFrameSuccess'));
      } finally {
        drawImageSpy.mockRestore();
      }
    } finally {
      coordinator.dispose();
    }
  });
});
