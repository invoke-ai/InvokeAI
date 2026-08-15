import type { QueueItem } from '@features/queue/core/historyTypes';
import type { QueueBackendInvocation, QueueBackendPort, QueueResultImage } from '@features/queue/core/types';

import { buildQueueItemOrigin } from '@features/queue/data/events';
import { describe, expect, it, vi } from 'vitest';

import { createQueueItemBackendSubmission, createQueueRuntime, type QueueHistoryCommands } from './runtime';

const createPendingQueueItem = (): QueueItem => ({
  cancellable: true,
  id: 'local-queue-item',
  snapshot: {
    backendSubmission: {
      batchCount: 1,
      graph: { edges: [], id: 'backend-graph', nodes: {} },
      kind: 'generate',
      negativePrompt: '',
      negativePromptNodeId: 'negative_prompt',
      positivePrompt: '',
      positivePromptNodeId: 'positive_prompt',
      seed: 0,
      seedNodeId: 'seed',
      shouldRandomizeSeed: false,
    },
    destination: 'canvas',
    filterIntermediateResults: false,
    galleryBoardId: null,
    graph: { id: 'graph', label: 'Generate' },
    presentation: { batchCount: 1, height: 1024, width: 1024 },
    resultNodeIds: ['canvas_output'],
    sourceId: 'generate',
    submittedAt: '2026-07-17T00:00:00.000Z',
  },
  status: 'pending',
});

describe('queue runtime', () => {
  it('turns a malformed legacy submission into a failed request instead of throwing', () => {
    const queueItem = createPendingQueueItem();
    delete (queueItem.snapshot as Partial<QueueItem['snapshot']>).backendSubmission;

    expect(createQueueItemBackendSubmission({ id: 'project-1' }, queueItem)).toEqual({
      error: 'Queue item is missing a compiled backend submission.',
      kind: 'invalid',
    });
  });

  it('adopts a persisted backend run before submission so reload cannot duplicate it', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const enqueueGenerate = vi.fn();
    const listItems = vi.fn().mockResolvedValue([
      {
        batchId: 'backend-batch',
        id: 77,
        origin: buildQueueItemOrigin(queueItem.id, project.id),
        status: 'in_progress',
      },
    ]);
    const backend: QueueBackendPort = {
      cancelCurrentItem: vi.fn(),
      cancelItem: vi.fn(),
      cancelQueueItems: vi.fn(),
      cancelQueueItemsByBatchIds: vi.fn(),
      cancelScopedItems: vi.fn(),
      clearFailedItems: vi.fn(),
      clearItems: vi.fn(),
      emit: vi.fn(),
      enqueueGenerate,
      enqueueWorkflow: vi.fn(),
      getItem: vi.fn(),
      getResultImages: vi.fn(),
      getResultVideoNames: vi.fn().mockResolvedValue([]),
      listItems,
      on: vi.fn(() => vi.fn()),
      onConnectionChange: vi.fn((listener) => {
        listener('connected');
        return vi.fn();
      }),
      pauseProcessor: vi.fn(),
      readCurrent: vi.fn(),
      readItemIds: vi.fn(),
      readItemsById: vi.fn(),
      readNext: vi.fn(),
      readStatus: vi.fn(),
      resumeProcessor: vi.fn(),
      retryItems: vi.fn(),
    };
    const commands: QueueHistoryCommands = {
      markBackendCancelled: vi.fn(),
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        queueItem.backendBatchId = backendBatchId;
        queueItem.backendItemIds = backendItemIds;
        queueItem.status = 'running';
      },
      recordError: vi.fn(),
      refreshBackendData: vi.fn(),
      routePartialResults: vi.fn(),
      routeResults: vi.fn(),
      setConnectionStatus: vi.fn(),
      setStatus: vi.fn(),
    };
    const runtime = createQueueRuntime({
      backend,
      destinations: { addImagesToGalleryBoard: vi.fn(), addVideosToGalleryBoard: vi.fn() },
      ensureTemplatesLoaded: vi.fn(),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      modelLoads: {
        completed: vi.fn(),
        reset: vi.fn(),
        started: vi.fn(),
      },
      nodeExecution: {
        clearAll: vi.fn(),
        completed: vi.fn(),
        failed: vi.fn(),
        progress: vi.fn(),
        settleRunning: vi.fn(),
        started: vi.fn(),
      },
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(queueItem).toMatchObject({
        backendBatchId: 'backend-batch',
        backendItemIds: [77],
        status: 'running',
      });
    });
    expect(listItems).toHaveBeenCalledTimes(1);
    expect(enqueueGenerate).not.toHaveBeenCalled();

    runtime.dispose();
  });
});

describe('queue runtime video board routing', () => {
  const createHarness = (options: {
    getResultVideoNames: QueueBackendPort['getResultVideoNames'];
    galleryBoardId?: string | null;
    getResultImages?: QueueBackendPort['getResultImages'];
    /** Nodes of the compiled submission graph — media values here mark run INPUTS. */
    graphNodes?: Record<string, QueueBackendInvocation>;
  }) => {
    const queueItem = createPendingQueueItem();
    queueItem.snapshot.destination = 'gallery';
    queueItem.snapshot.galleryBoardId = options.galleryBoardId === undefined ? 'board-1' : options.galleryBoardId;
    queueItem.snapshot.filterIntermediateResults = true;
    delete (queueItem.snapshot as { resultNodeIds?: unknown }).resultNodeIds;
    if (options.graphNodes && queueItem.snapshot.backendSubmission.kind !== 'invalid') {
      queueItem.snapshot.backendSubmission.graph.nodes = options.graphNodes;
    }
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const backend: QueueBackendPort = {
      cancelCurrentItem: vi.fn(),
      cancelItem: vi.fn(),
      cancelQueueItems: vi.fn(),
      cancelQueueItemsByBatchIds: vi.fn(),
      cancelScopedItems: vi.fn(),
      clearFailedItems: vi.fn(),
      clearItems: vi.fn(),
      emit: vi.fn(),
      enqueueGenerate: vi.fn(),
      enqueueWorkflow: vi.fn(),
      getItem: vi.fn(),
      getResultImages: options.getResultImages ?? vi.fn().mockResolvedValue([]),
      getResultVideoNames: options.getResultVideoNames,
      // The backend already accepted and completed this run before "reload": reconcile
      // adopts it and settles immediately, driving both settlement paths without sockets.
      listItems: vi.fn().mockResolvedValue([
        {
          batchId: 'backend-batch',
          id: 77,
          origin: buildQueueItemOrigin(queueItem.id, project.id),
          status: 'completed',
        },
      ]),
      on: vi.fn(() => vi.fn()),
      onConnectionChange: vi.fn(() => vi.fn()),
      pauseProcessor: vi.fn(),
      readCurrent: vi.fn().mockResolvedValue(null),
      readItemIds: vi.fn().mockResolvedValue({ itemIds: [], totalCount: 0 }),
      readItemsById: vi.fn().mockResolvedValue([]),
      readNext: vi.fn().mockResolvedValue(null),
      readStatus: vi.fn().mockResolvedValue({
        processor: { isProcessing: false, isStarted: true },
        queue: {
          canceled: 0,
          completed: 0,
          failed: 0,
          inProgress: 0,
          pending: 0,
          queueId: 'default',
          total: 0,
          waiting: 0,
        },
      }),
      resumeProcessor: vi.fn(),
      retryItems: vi.fn(),
    };
    const commands: QueueHistoryCommands = {
      markBackendCancelled: vi.fn(),
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        queueItem.backendBatchId = backendBatchId;
        queueItem.backendItemIds = backendItemIds;
        queueItem.status = 'running';
      },
      recordError: vi.fn(),
      refreshBackendData: vi.fn(),
      routePartialResults: vi.fn(),
      routeResults: vi.fn(),
      setConnectionStatus: vi.fn(),
      setStatus: vi.fn(),
    };
    const destinations = { addImagesToGalleryBoard: vi.fn(), addVideosToGalleryBoard: vi.fn() };
    const runtime = createQueueRuntime({
      backend,
      destinations,
      ensureTemplatesLoaded: vi.fn(),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      modelLoads: { completed: vi.fn(), reset: vi.fn(), started: vi.fn() },
      nodeExecution: {
        clearAll: vi.fn(),
        completed: vi.fn(),
        failed: vi.fn(),
        progress: vi.fn(),
        settleRunning: vi.fn(),
        started: vi.fn(),
      },
    });

    return { commands, destinations, runtime };
  };

  it('lands result videos on the enqueue-time board, excluding intermediates', async () => {
    const getResultVideoNames = vi.fn().mockResolvedValue(['clip-1.mp4']);
    const { destinations, runtime } = createHarness({ getResultVideoNames });

    runtime.start();

    await vi.waitFor(() => {
      expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledWith('board-1', ['clip-1.mp4']);
    });
    // filterIntermediateResults on the snapshot maps to the video-side intermediate filter.
    expect(getResultVideoNames).toHaveBeenCalledWith(77, expect.objectContaining({ excludeIntermediate: true }));

    runtime.dispose();
  });

  it('records a video-routing failure without failing the completed run', async () => {
    const getResultVideoNames = vi.fn().mockRejectedValue(new Error('transient 502'));
    const { commands, destinations, runtime } = createHarness({ getResultVideoNames });

    runtime.start();

    await vi.waitFor(() => {
      // The run still settles and records its results...
      expect(commands.routeResults).toHaveBeenCalled();
      // ...with the board-attach hiccup recorded, not fatal.
      expect(commands.recordError).toHaveBeenCalledWith(expect.objectContaining({ area: 'queue-results' }));
    });
    expect(commands.setStatus).not.toHaveBeenCalledWith(expect.objectContaining({ status: 'failed' }));
    expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalled();

    runtime.dispose();
  });

  it('skips the board attach entirely when no board was active at enqueue', async () => {
    const getResultVideoNames = vi.fn().mockResolvedValue(['clip-1.mp4']);
    const { commands, destinations, runtime } = createHarness({ galleryBoardId: null, getResultVideoNames });

    runtime.start();

    // Wait for full settlement first so the negative assertions below are meaningful.
    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalled();
    });
    expect(destinations.addImagesToGalleryBoard).not.toHaveBeenCalled();
    expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalled();
    expect(getResultVideoNames).not.toHaveBeenCalled();

    runtime.dispose();
  });

  const resultImage = (imageName: string): QueueResultImage => ({
    height: 512,
    imageName,
    imageUrl: `http://test/i/${imageName}`,
    isIntermediate: false,
    queuedAt: '2026-07-17T00:00:00.000Z',
    sourceQueueItemId: 'local-queue-item',
    thumbnailUrl: `http://test/t/${imageName}`,
    width: 512,
  });

  it('never routes an input image echoed into the results (first-frame keyframe)', async () => {
    // The i2v workflow's `image` primitive echoes the uploaded keyframe into
    // session.results as a non-intermediate output; only the generated image may
    // reach the board or the recorded results.
    const { commands, destinations, runtime } = createHarness({
      getResultImages: vi.fn().mockResolvedValue([resultImage('keyframe.png'), resultImage('generated.png')]),
      getResultVideoNames: vi.fn().mockResolvedValue([]),
      graphNodes: {
        first_frame: { id: 'first_frame', image: { image_name: 'keyframe.png' }, type: 'image' },
      },
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(destinations.addImagesToGalleryBoard).toHaveBeenCalledWith('board-1', ['generated.png']);
      expect(commands.routeResults).toHaveBeenCalledWith(
        expect.objectContaining({ images: [expect.objectContaining({ imageName: 'generated.png' })] })
      );
    });
    expect(destinations.addImagesToGalleryBoard).not.toHaveBeenCalledWith(
      'board-1',
      expect.arrayContaining(['keyframe.png'])
    );

    runtime.dispose();
  });

  it('never routes an input video echoed into the results (extend-video source clip)', async () => {
    const { destinations, runtime } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['source.mp4', 'extended.mp4']),
      graphNodes: {
        source_clip: { id: 'source_clip', type: 'video', video: { video_name: 'source.mp4' } },
      },
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledWith('board-1', ['extended.mp4']);
    });
    expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalledWith(
      'board-1',
      expect.arrayContaining(['source.mp4'])
    );

    runtime.dispose();
  });
});
