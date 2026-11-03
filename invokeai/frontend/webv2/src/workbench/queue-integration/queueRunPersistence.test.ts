import type { WorkbenchQueueItem } from '@workbench/queueHistoryContracts';

import { createDefaultVideoWidgetValues } from '@features/video';
import { expect, it, vi } from 'vitest';

import { withQueueRecallCache } from './queueRunPersistence';

it('retains exact recall when a terminal item disappears before journal settlement', async () => {
  const put = vi.fn().mockResolvedValue({ kind: 'stored' });
  const settle = vi.fn().mockResolvedValue({ kind: 'removed' });
  const journal = { record: vi.fn().mockResolvedValue({ kind: 'stored' }), settle };
  const videoValues = createDefaultVideoWidgetValues();
  const item = {
    id: 'q',
    status: 'pending',
    snapshot: {
      sourceId: 'video',
      submittedAt: '2026-09-04T00:00:00Z',
      recall: { videoValues },
    },
  } as WorkbenchQueueItem;
  const persistence = withQueueRecallCache(journal, { put }, () => undefined);
  await persistence.record('p', item);
  await persistence.settle('p', 'q');
  expect(put).toHaveBeenCalledWith({
    projectId: 'p',
    queueItemId: 'q',
    sourceId: 'video',
    submittedAt: '2026-09-04T00:00:00Z',
    videoValues,
  });
  expect(settle).toHaveBeenCalledWith('p', 'q');
});

it('does not prevent run cleanup when the optional recall cache fails', async () => {
  const settle = vi.fn().mockResolvedValue({ kind: 'removed' });
  const item = {
    id: 'q',
    status: 'failed',
    snapshot: {
      sourceId: 'video',
      submittedAt: '2026-09-04T00:00:00Z',
      recall: { videoValues: createDefaultVideoWidgetValues() },
    },
  } as WorkbenchQueueItem;
  const persistence = withQueueRecallCache(
    { record: vi.fn(), settle },
    { put: vi.fn().mockRejectedValue(new Error('quota')) },
    () => item
  );
  await expect(persistence.settle('p', 'q')).resolves.toEqual({ kind: 'removed' });
});
