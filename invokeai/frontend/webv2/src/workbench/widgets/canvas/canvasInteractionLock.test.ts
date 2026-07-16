import type { CanvasStagingCandidateContract, QueueItem, QueueItemStatus } from '@workbench/types';

import { createEmptyCanvasStateV2 } from '@workbench/canvasMigration';
import { describe, expect, it } from 'vitest';

import {
  getCanvasInteractionCapabilities,
  isCanvasInteractionLocked,
  isCanvasStagingActive,
  isCanvasToolEnabled,
} from './canvasInteractionLock';

const createStagedCandidate = (): CanvasStagingCandidateContract => ({
  height: 512,
  imageName: 'staged.png',
  imageUrl: '/api/v1/images/i/staged.png/full',
  placement: { height: 512, opacity: 1, width: 512, x: 0, y: 0 },
  queuedAt: '2026-06-09T00:00:00.000Z',
  sourceQueueItemId: 'queue-staged',
  thumbnailUrl: '/api/v1/images/i/staged.png/thumbnail',
  width: 512,
});

const createCanvas = ({ revision = 1, staged = false }: { revision?: number; staged?: boolean } = {}) => {
  const canvas = createEmptyCanvasStateV2(512, 512);
  const pendingImages = staged ? [createStagedCandidate()] : [];

  return {
    ...canvas,
    documentRevision: revision,
    stagingArea: {
      ...canvas.stagingArea,
      isVisible: staged,
      pendingImageIds: pendingImages.map((image) => image.imageName),
      pendingImages,
    },
  };
};

const createQueueItem = ({ revision = 1, status }: { revision?: number; status: QueueItemStatus }): QueueItem =>
  ({
    backendItemIds: [1],
    completedBackendItemIds: [1],
    id: `queue-${status}`,
    snapshot: {
      canvas: { documentRevision: revision },
      destination: 'canvas',
      submittedAt: '2026-06-09T00:00:00.000Z',
    },
    status,
  }) as QueueItem;

describe('canvas interaction lock', () => {
  it.each([
    {
      expected: {
        areOperationActionsEnabled: false,
        canAcceptStagedImage: false,
        isDocumentEditingLocked: false,
        isOperationChromeVisible: false,
        isRegularToolOptionsVisible: true,
        isSurfaceInteractionLocked: false,
      },
      generation: false,
      operationKind: null,
      staged: false,
    },
    {
      expected: {
        areOperationActionsEnabled: true,
        canAcceptStagedImage: false,
        isDocumentEditingLocked: true,
        isOperationChromeVisible: true,
        isRegularToolOptionsVisible: false,
        isSurfaceInteractionLocked: false,
      },
      generation: false,
      operationKind: 'filter' as const,
      staged: false,
    },
    {
      expected: {
        areOperationActionsEnabled: true,
        canAcceptStagedImage: false,
        isDocumentEditingLocked: true,
        isOperationChromeVisible: true,
        isRegularToolOptionsVisible: false,
        isSurfaceInteractionLocked: false,
      },
      generation: false,
      operationKind: 'select-object' as const,
      staged: false,
    },
    {
      expected: {
        areOperationActionsEnabled: false,
        canAcceptStagedImage: true,
        isDocumentEditingLocked: false,
        isOperationChromeVisible: false,
        isRegularToolOptionsVisible: false,
        isSurfaceInteractionLocked: true,
      },
      generation: false,
      operationKind: null,
      staged: true,
    },
    {
      expected: {
        areOperationActionsEnabled: false,
        canAcceptStagedImage: false,
        isDocumentEditingLocked: false,
        isOperationChromeVisible: false,
        isRegularToolOptionsVisible: false,
        isSurfaceInteractionLocked: true,
      },
      generation: true,
      operationKind: null,
      staged: false,
    },
    {
      expected: {
        areOperationActionsEnabled: false,
        canAcceptStagedImage: false,
        isDocumentEditingLocked: true,
        isOperationChromeVisible: true,
        isRegularToolOptionsVisible: false,
        isSurfaceInteractionLocked: true,
      },
      generation: false,
      operationKind: 'filter' as const,
      staged: true,
    },
    {
      expected: {
        areOperationActionsEnabled: false,
        canAcceptStagedImage: false,
        isDocumentEditingLocked: true,
        isOperationChromeVisible: true,
        isRegularToolOptionsVisible: false,
        isSurfaceInteractionLocked: true,
      },
      generation: true,
      operationKind: 'select-object' as const,
      staged: false,
    },
  ])(
    'separates surface, document, and chrome capabilities for operation=$operationKind staged=$staged generation=$generation',
    ({ expected, generation, operationKind, staged }) => {
      expect(
        getCanvasInteractionCapabilities({
          hasCanvasEngine: true,
          hasSelectedCandidate: staged,
          hasStagingSlots: staged,
          isCanvasGenerationInFlight: generation,
          operationKind,
        })
      ).toEqual(expected);
    }
  );

  it('does not lock an empty canvas', () => {
    expect(isCanvasInteractionLocked(createCanvas(), [])).toBe(false);
  });

  it('locks while a staged candidate exists', () => {
    expect(isCanvasInteractionLocked(createCanvas({ staged: true }), [])).toBe(true);
  });

  it('locks for a pending queue item on the current document revision without staging slots', () => {
    expect(
      isCanvasInteractionLocked(createCanvas({ revision: 2 }), [createQueueItem({ revision: 2, status: 'pending' })])
    ).toBe(true);
  });

  it('locks for a running queue item on the current document revision without staging slots', () => {
    expect(
      isCanvasInteractionLocked(createCanvas({ revision: 2 }), [createQueueItem({ revision: 2, status: 'running' })])
    ).toBe(true);
  });

  it('does not lock for terminal queue items', () => {
    const canvas = createCanvas({ revision: 2 });

    for (const status of ['completed', 'failed', 'cancelled'] as const) {
      expect(isCanvasInteractionLocked(canvas, [createQueueItem({ revision: 2, status })])).toBe(false);
    }
  });

  it('does not lock for an active queue item from a stale document revision', () => {
    expect(
      isCanvasInteractionLocked(createCanvas({ revision: 2 }), [createQueueItem({ revision: 1, status: 'running' })])
    ).toBe(false);
  });

  it('treats pending staged candidates and in-flight canvas generation as active staging', () => {
    expect(isCanvasStagingActive({ hasStagedCandidates: true, isCanvasGenerationInFlight: false })).toBe(true);
    expect(isCanvasStagingActive({ hasStagedCandidates: false, isCanvasGenerationInFlight: true })).toBe(true);
    expect(isCanvasStagingActive({ hasStagedCandidates: false, isCanvasGenerationInFlight: false })).toBe(false);
  });

  it('allows only the view tool while staging locks canvas interactions', () => {
    expect(isCanvasToolEnabled('view', true)).toBe(true);
    expect(isCanvasToolEnabled('bbox', true)).toBe(false);
    expect(isCanvasToolEnabled('brush', true)).toBe(false);
    expect(isCanvasToolEnabled('move', true)).toBe(false);
    expect(isCanvasToolEnabled('colorPicker', true)).toBe(false);
  });

  it('allows all tools when staging is inactive', () => {
    expect(isCanvasToolEnabled('bbox', false)).toBe(true);
    expect(isCanvasToolEnabled('brush', false)).toBe(true);
    expect(isCanvasToolEnabled('colorPicker', false)).toBe(true);
  });
});
