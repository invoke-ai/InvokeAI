import { describe, expect, it } from 'vitest';

import type { QueueItem } from './historyTypes';
import type { QueueItemReadModel } from './types';

import { getQueueItemReadModelLabel, getQueueItemSnapshotLabel, shortenQueueLabel } from './itemLabels';

const readModel = (overrides: Partial<QueueItemReadModel>): QueueItemReadModel =>
  ({ fieldValues: [], id: 7, ...overrides }) as QueueItemReadModel;

const snapshotItem = (backendSubmission: object, positivePrompt?: string): QueueItem =>
  ({ snapshot: { backendSubmission, presentation: { positivePrompt } } }) as unknown as QueueItem;

describe('queue item labels', () => {
  it('names a backend item by its workflow, else by its prompt on one line', () => {
    const prompt = { fieldName: 'value', nodePath: 'positive_prompt', value: 'a fox\n  in the snow' };

    expect(getQueueItemReadModelLabel(readModel({ fieldValues: [prompt], workflowName: 'Portrait upscale' }))).toBe(
      'Portrait upscale'
    );
    expect(getQueueItemReadModelLabel(readModel({ fieldValues: [prompt] }))).toBe('a fox in the snow');
    expect(getQueueItemReadModelLabel(readModel({}))).toBe('');
  });

  it('names a submitted run by its workflow, else by its prompt', () => {
    expect(getQueueItemSnapshotLabel(snapshotItem({ kind: 'workflow', workflow: { name: 'Tiled upscale' } }))).toBe(
      'Tiled upscale'
    );
    expect(getQueueItemSnapshotLabel(snapshotItem({ kind: 'generate' }, 'A castle at dusk'))).toBe('A castle at dusk');
    expect(getQueueItemSnapshotLabel(snapshotItem({ kind: 'workflow' }))).toBe('');
  });

  it('cuts long labels at a word with an ellipsis', () => {
    const label = shortenQueueLabel('a very long prompt about a lighthouse standing in a storm at night', 30);

    expect(label).toBe('a very long prompt about a…');
    expect(label.length).toBeLessThanOrEqual(30);
  });
});
