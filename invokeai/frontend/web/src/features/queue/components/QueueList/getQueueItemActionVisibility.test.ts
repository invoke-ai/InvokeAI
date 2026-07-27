import type { S } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { getQueueItemActionVisibility } from './getQueueItemActionVisibility';

const buildQueueItem = (overrides: Partial<S['SessionQueueItem']> = {}): S['SessionQueueItem'] =>
  ({
    item_id: 1,
    status: 'pending',
    priority: 0,
    batch_id: 'batch-1',
    origin: null,
    destination: null,
    session_id: 'session-1',
    error_type: null,
    error_message: null,
    error_traceback: null,
    created_at: '2026-04-21T00:00:00Z',
    updated_at: '2026-04-21T00:00:00Z',
    started_at: null,
    completed_at: null,
    queue_id: 'default',
    user_id: 'user-1',
    user_display_name: null,
    user_email: null,
    field_values: null,
    retried_from_item_id: null,
    workflow_call_id: null,
    parent_item_id: null,
    parent_session_id: null,
    root_item_id: null,
    workflow_call_depth: null,
    session: {
      id: 'ges-1',
      graph: { id: 'graph-1', nodes: {}, edges: [] },
      execution_graph: { id: 'graph-2', nodes: {}, edges: [] },
      executed: [],
      executed_history: [],
      results: {},
      errors: {},
      workflow_call_stack: [],
      workflow_call_history: [],
      prepared_source_mapping: {},
      source_prepared_mapping: {},
    },
    workflow: null,
    ...overrides,
  }) as S['SessionQueueItem'];

describe('getQueueItemActionVisibility', () => {
  it('shows cancel and retry for root queue items', () => {
    const queueItem = buildQueueItem();

    expect(getQueueItemActionVisibility(queueItem).canShowCancelQueueItem).toBe(true);
    expect(getQueueItemActionVisibility(queueItem).canShowRetryQueueItem).toBe(true);
  });

  it('shows cancel and hides retry for child queue items', () => {
    const queueItem = buildQueueItem({ parent_item_id: 42, workflow_call_id: 'workflow-call-1' });

    expect(getQueueItemActionVisibility(queueItem).canShowCancelQueueItem).toBe(true);
    expect(getQueueItemActionVisibility(queueItem).canShowRetryQueueItem).toBe(false);
  });
});
