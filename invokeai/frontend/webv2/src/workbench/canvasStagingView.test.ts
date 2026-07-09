import { describe, expect, it } from 'vitest';

import type { CanvasStagingCandidateContract, GeneratedImageContract, QueueItem, QueueItemStatus } from './types';

import { createEmptyCanvasStateV2 } from './canvasMigration';
import { getCanvasQueuePlaceholderSlots, getCanvasStagingSlots } from './canvasStagingView';

const createImage = (imageName: string, sourceQueueItemId: string): GeneratedImageContract => ({
  height: 768,
  imageName,
  imageUrl: `/api/v1/images/i/${imageName}/full`,
  queuedAt: '2026-06-09T00:00:00.000Z',
  sourceQueueItemId,
  thumbnailUrl: `/api/v1/images/i/${imageName}/thumbnail`,
  width: 512,
});

const createCandidate = (imageName: string, sourceQueueItemId: string): CanvasStagingCandidateContract => ({
  ...createImage(imageName, sourceQueueItemId),
  placement: { height: 256, opacity: 1, width: 256, x: 16, y: 24 },
});

const createBackendCandidate = (
  imageName: string,
  sourceQueueItemId: string,
  sourceBackendItemId: number
): CanvasStagingCandidateContract => ({
  ...createCandidate(imageName, sourceQueueItemId),
  sourceBackendItemId,
});

const createCanvas = ({
  bbox = { height: 512, width: 512, x: 0, y: 0 },
  revision = 0,
  stagedImages = [],
}: {
  bbox?: { height: number; width: number; x: number; y: number };
  revision?: number;
  stagedImages?: CanvasStagingCandidateContract[];
} = {}) => {
  const canvas = createEmptyCanvasStateV2(1024, 1024);

  return {
    ...canvas,
    document: { ...canvas.document, bbox },
    documentRevision: revision,
    stagingArea: {
      ...canvas.stagingArea,
      isVisible: stagedImages.length > 0,
      pendingImageIds: stagedImages.map((image) => image.imageName),
      pendingImages: stagedImages,
    },
  };
};

const createQueueItem = ({
  backendItemIds,
  batchCount = 1,
  bbox = { height: 320, width: 640, x: 10, y: 12 },
  cancelledBackendItemIds,
  completedBackendItemIds,
  destination = 'canvas',
  id,
  revision = 0,
  status = 'pending',
  submittedAt = '2026-06-09T00:00:00.000Z',
}: {
  backendItemIds?: number[];
  batchCount?: number;
  bbox?: { height: number; width: number; x: number; y: number };
  cancelledBackendItemIds?: number[];
  completedBackendItemIds?: number[];
  destination?: 'canvas' | 'gallery';
  id: string;
  revision?: number;
  status?: QueueItemStatus;
  submittedAt?: string;
}): QueueItem =>
  ({
    backendItemIds,
    cancellable: true,
    cancelledBackendItemIds,
    completedBackendItemIds,
    id,
    snapshot: {
      canvas: createCanvas({ bbox, revision }),
      destination,
      graph: { edges: [], id: 'graph', label: 'Graph', nodes: [], updatedAt: '2026-06-09T00:00:00.000Z', version: 1 },
      sourceId: 'canvas',
      submittedAt,
      widgetInstances: {},
      widgetStates: {
        generate: { id: 'generate', label: 'Generate', values: { batchCount, height: 999, width: 999 }, version: 1 },
      },
    },
    status,
  }) as QueueItem;

describe('canvas staging view', () => {
  it('derives in-flight canvas placeholders from the submitted bbox without persisting fake images', () => {
    const canvas = createCanvas({ revision: 2 });
    const queueItems = [
      createQueueItem({
        batchCount: 2,
        bbox: { height: 384, width: 640, x: 8, y: 9 },
        id: 'q-pending',
        revision: 2,
        submittedAt: '2026-06-09T00:00:01.000Z',
      }),
      createQueueItem({
        backendItemIds: [11, 12, 13],
        bbox: { height: 256, width: 512, x: 1, y: 2 },
        cancelledBackendItemIds: [12],
        completedBackendItemIds: [11],
        id: 'q-running',
        revision: 2,
        status: 'running',
        submittedAt: '2026-06-09T00:00:02.000Z',
      }),
    ];

    expect(getCanvasQueuePlaceholderSlots(canvas, queueItems)).toEqual([
      { height: 384, id: 'q-pending:0', itemIndex: 1, kind: 'placeholder', queueItemId: 'q-pending', width: 640 },
      { height: 384, id: 'q-pending:1', itemIndex: 2, kind: 'placeholder', queueItemId: 'q-pending', width: 640 },
      { height: 256, id: 'q-running:2', itemIndex: 3, kind: 'placeholder', queueItemId: 'q-running', width: 512 },
    ]);
    expect(canvas.stagingArea.pendingImages).toEqual([]);
  });

  it('skips gallery, stale, terminal, completed, and cancelled canvas slots', () => {
    const canvas = createCanvas({ revision: 2 });
    const queueItems = [
      createQueueItem({ destination: 'gallery', id: 'q-gallery', revision: 2 }),
      createQueueItem({ id: 'q-stale', revision: 1 }),
      createQueueItem({ id: 'q-completed', revision: 2, status: 'completed' }),
      createQueueItem({ id: 'q-cancelled', revision: 2, status: 'cancelled' }),
      createQueueItem({
        backendItemIds: [11, 12],
        cancelledBackendItemIds: [12],
        completedBackendItemIds: [11],
        id: 'q-no-open-slots',
        revision: 2,
        status: 'running',
      }),
    ];

    expect(getCanvasQueuePlaceholderSlots(canvas, queueItems)).toEqual([]);
  });

  it('combines ready staged candidates with ephemeral placeholders in one selected slot list', () => {
    const staged = createCandidate('ready.png', 'q-ready');
    const canvas = createCanvas({ revision: 2, stagedImages: [staged] });
    const queueItems = [createQueueItem({ id: 'q-pending', revision: 2 })];

    expect(getCanvasStagingSlots(canvas, queueItems)).toEqual([
      {
        candidate: staged,
        height: 768,
        id: 'candidate:q-ready:ready.png',
        imageName: 'ready.png',
        kind: 'candidate',
        queueItemId: 'q-ready',
        width: 512,
      },
      { height: 320, id: 'q-pending:0', itemIndex: 1, kind: 'placeholder', queueItemId: 'q-pending', width: 640 },
    ]);
  });

  it('orders placeholders by execution order even when local queue items are newest-first', () => {
    const canvas = createCanvas({ revision: 2 });
    const queueItems = [
      createQueueItem({ id: 'q-third', revision: 2, submittedAt: '2026-06-09T00:00:03.000Z' }),
      createQueueItem({ id: 'q-second', revision: 2, submittedAt: '2026-06-09T00:00:02.000Z' }),
      createQueueItem({ id: 'q-first', revision: 2, submittedAt: '2026-06-09T00:00:01.000Z' }),
    ];

    expect(getCanvasQueuePlaceholderSlots(canvas, queueItems).map((slot) => slot.queueItemId)).toEqual([
      'q-first',
      'q-second',
      'q-third',
    ]);
  });

  it('interleaves completed candidates and placeholders per queue item in execution order', () => {
    const firstCandidate = createBackendCandidate('first-result.png', 'q-first', 11);
    const canvas = createCanvas({ revision: 2, stagedImages: [firstCandidate] });
    const queueItems = [
      createQueueItem({ id: 'q-second', revision: 2, submittedAt: '2026-06-09T00:00:02.000Z' }),
      createQueueItem({
        backendItemIds: [11, 12],
        completedBackendItemIds: [11],
        id: 'q-first',
        revision: 2,
        status: 'running',
        submittedAt: '2026-06-09T00:00:01.000Z',
      }),
    ];

    expect(
      getCanvasStagingSlots(canvas, queueItems).map((slot) =>
        slot.kind === 'candidate'
          ? `candidate:${slot.queueItemId}:${slot.itemIndex}`
          : `placeholder:${slot.queueItemId}:${slot.itemIndex}`
      )
    ).toEqual(['candidate:q-first:1', 'placeholder:q-first:2', 'placeholder:q-second:1']);
  });
});
