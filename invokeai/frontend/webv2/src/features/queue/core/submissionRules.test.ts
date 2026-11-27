import type { QueueHistoryItemStatus, QueueItem } from '@features/queue';
import type { QueueSourceId } from '@features/queue/core/types';

import { describe, expect, it } from 'vitest';

import { isBackendSubmittableSourceId, shouldSubmitPendingQueueItem } from './submissionRules';

/** Minimal queue item whose submission gate only reads `status` + `snapshot.sourceId`. */
const makeQueueItem = (sourceId: QueueSourceId, status: QueueHistoryItemStatus): QueueItem =>
  ({
    id: 'queue-item-1',
    status,
    cancellable: true,
    snapshot: { sourceId },
  }) as unknown as QueueItem;

describe('isBackendSubmittableSourceId', () => {
  it('includes canvas so canvas invocations are actually enqueued (regression: canvas→canvas stall)', () => {
    // Canvas source IDs must be allowed or local pending items never submit.
    expect(isBackendSubmittableSourceId('canvas')).toBe(true);
  });

  it.each(['generate', 'workflow', 'upscale'] as const)('submits %s invocations to the backend', (sourceId) => {
    expect(isBackendSubmittableSourceId(sourceId)).toBe(true);
  });
});

describe('shouldSubmitPendingQueueItem', () => {
  it.each<[QueueSourceId, QueueHistoryItemStatus, boolean]>([
    ['canvas', 'pending', true],
    ['generate', 'pending', true],
    ['upscale', 'pending', true],
    ['canvas', 'running', false],
    ['canvas', 'completed', false],
    ['canvas', 'cancelled', false],
  ])('decides a %s item in %s submits: %s', (sourceId, status, expected) => {
    expect(shouldSubmitPendingQueueItem(makeQueueItem(sourceId, status))).toBe(expected);
  });
});
