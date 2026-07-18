import { describe, expect, it } from 'vitest';

import type { InvocationSourceId, QueueItem, QueueItemStatus } from './types';

import {
  BACKEND_SUBMITTABLE_SOURCE_IDS,
  isBackendSubmittableSourceId,
  shouldSubmitPendingQueueItem,
} from './queueSubmission';

/** Minimal queue item whose submission gate only reads `status` + `snapshot.sourceId`. */
const makeQueueItem = (sourceId: InvocationSourceId, status: QueueItemStatus): QueueItem =>
  ({
    id: 'queue-item-1',
    status,
    cancellable: true,
    snapshot: { sourceId },
  }) as unknown as QueueItem;

describe('isBackendSubmittableSourceId', () => {
  it('includes canvas so canvas invocations are actually enqueued (regression: canvas→canvas stall)', () => {
    // The bug: canvas snapshots carry sourceId 'canvas'; the runtime allow-list
    // only had 'generate'/'workflow', so canvas items stacked as local pending
    // rows forever and nothing generated.
    expect(isBackendSubmittableSourceId('canvas')).toBe(true);
  });

  it('includes generate and workflow', () => {
    expect(isBackendSubmittableSourceId('generate')).toBe(true);
    expect(isBackendSubmittableSourceId('workflow')).toBe(true);
  });

  it('includes upscale', () => {
    expect(isBackendSubmittableSourceId('upscale')).toBe(true);
  });

  it('exposes canvas in the exported allow-list', () => {
    expect(BACKEND_SUBMITTABLE_SOURCE_IDS).toContain('canvas');
  });
});

describe('shouldSubmitPendingQueueItem', () => {
  it('submits a pending canvas item', () => {
    expect(shouldSubmitPendingQueueItem(makeQueueItem('canvas', 'pending'))).toBe(true);
  });

  it('submits a pending generate item', () => {
    expect(shouldSubmitPendingQueueItem(makeQueueItem('generate', 'pending'))).toBe(true);
  });

  it('does not submit a non-pending canvas item', () => {
    expect(shouldSubmitPendingQueueItem(makeQueueItem('canvas', 'running'))).toBe(false);
    expect(shouldSubmitPendingQueueItem(makeQueueItem('canvas', 'completed'))).toBe(false);
    expect(shouldSubmitPendingQueueItem(makeQueueItem('canvas', 'cancelled'))).toBe(false);
  });

  it('submits a pending upscale item', () => {
    expect(shouldSubmitPendingQueueItem(makeQueueItem('upscale', 'pending'))).toBe(true);
  });
});
