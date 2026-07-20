import { describe, expect, it } from 'vitest';

import { mapQueueItemDTO, mapQueueItemIdsDTO, mapQueueStatusDTO } from './mappers';

describe('queue transport mappers', () => {
  it('maps a backend queue item into the live read model', () => {
    expect(
      mapQueueItemDTO({
        batch_id: 'batch-1',
        completed_at: '2026-07-18T01:00:00Z',
        created_at: '2026-07-18T00:00:00Z',
        destination: 'gallery',
        error_message: null,
        error_traceback: null,
        error_type: null,
        field_values: [
          { field_name: 'seed', node_path: 'noise', value: 42 },
          { field_name: 'image', node_path: 'input', value: { image_name: 'input.png' } },
        ],
        item_id: 7,
        origin: 'webv2:item-1',
        retried_from_item_id: 3,
        session: {
          results: {
            collection: { collection: [{ image_name: 'first.png' }, { image_name: 'second.png' }] },
            image: { image: { image_name: 'single.png' } },
          },
        },
        session_id: 'session-1',
        started_at: '2026-07-18T00:30:00Z',
        status: 'completed',
        updated_at: '2026-07-18T01:00:00Z',
      })
    ).toEqual({
      batchId: 'batch-1',
      completedAt: '2026-07-18T01:00:00Z',
      createdAt: '2026-07-18T00:00:00Z',
      destination: 'gallery',
      errorMessage: null,
      errorTraceback: null,
      errorType: null,
      fieldValues: [
        { fieldName: 'seed', nodePath: 'noise', value: 42 },
        { fieldName: 'image', nodePath: 'input', value: { imageName: 'input.png' } },
      ],
      id: 7,
      origin: 'webv2:item-1',
      resultImageNames: ['first.png', 'second.png', 'single.png'],
      retriedFromItemId: 3,
      sessionId: 'session-1',
      startedAt: '2026-07-18T00:30:00Z',
      status: 'completed',
      updatedAt: '2026-07-18T01:00:00Z',
    });
  });

  it('preserves absent nullable fields without exposing the raw session', () => {
    const item = mapQueueItemDTO({
      batch_id: 'batch-2',
      created_at: '2026-07-18T00:00:00Z',
      item_id: 8,
      session_id: 'session-2',
      status: 'pending',
      updated_at: '2026-07-18T00:00:00Z',
    });

    expect(item.fieldValues).toBeUndefined();
    expect(item.resultImageNames).toEqual([]);
    expect(item).not.toHaveProperty('session');
  });

  it('maps status and id-list response names at the adapter boundary', () => {
    expect(mapQueueItemIdsDTO({ item_ids: [9, 8], total_count: 2 })).toEqual({ itemIds: [9, 8], totalCount: 2 });
    expect(
      mapQueueStatusDTO({
        processor: { is_processing: true, is_started: true },
        queue: {
          batch_id: 'batch-1',
          canceled: 1,
          completed: 2,
          failed: 3,
          in_progress: 4,
          item_id: 7,
          pending: 5,
          queue_id: 'default',
          session_id: 'session-1',
          total: 15,
        },
      })
    ).toEqual({
      processor: { isProcessing: true, isStarted: true },
      queue: {
        batchId: 'batch-1',
        canceled: 1,
        completed: 2,
        failed: 3,
        inProgress: 4,
        itemId: 7,
        pending: 5,
        queueId: 'default',
        sessionId: 'session-1',
        total: 15,
      },
    });
  });
});
