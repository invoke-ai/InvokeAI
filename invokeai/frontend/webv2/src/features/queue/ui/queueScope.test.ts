import type { QueueItem, QueueItemReadModel } from '@features/queue';

import { buildQueueItemOrigin, buildProjectQueueItemOriginPrefix } from '@features/queue';
import { describe, expect, it } from 'vitest';

import { getQueueQueryScope, isQueueItemReadModelInProject } from './queueScope';

const createProject = (items: QueueItem[]) => ({ id: 'project-1', name: 'Project 1', queue: { items } });

const createLocalQueueItem = ({
  backendItemIds,
  id,
  status,
}: {
  backendItemIds?: number[];
  id: string;
  status: QueueItem['status'];
}): QueueItem =>
  ({
    backendItemIds,
    id,
    status,
  }) as QueueItem;

const createServerQueueItem = ({ itemId, origin }: { itemId: number; origin?: string | null }): QueueItemReadModel =>
  ({
    id: itemId,
    origin,
  }) as QueueItemReadModel;

describe('isQueueItemReadModelInProject', () => {
  it('matches server queue items by webv2 origin', () => {
    const project = createProject([createLocalQueueItem({ id: 'local-1', status: 'pending' })]);

    expect(
      isQueueItemReadModelInProject(
        createServerQueueItem({ itemId: 7, origin: buildQueueItemOrigin('local-1', 'project-1') }),
        project
      )
    ).toBe(true);
    expect(isQueueItemReadModelInProject(createServerQueueItem({ itemId: 7, origin: 'webv2:local-1' }), project)).toBe(
      true
    );
    expect(isQueueItemReadModelInProject(createServerQueueItem({ itemId: 8, origin: 'webv2:local-2' }), project)).toBe(
      false
    );
  });

  it('falls back to backend item ids when origin is unavailable', () => {
    const project = createProject([createLocalQueueItem({ backendItemIds: [7, 8], id: 'local-1', status: 'running' })]);

    expect(isQueueItemReadModelInProject(createServerQueueItem({ itemId: 8, origin: null }), project)).toBe(true);
    expect(isQueueItemReadModelInProject(createServerQueueItem({ itemId: 9, origin: null }), project)).toBe(false);
  });
  it('builds backend origin-prefix scopes from the active project', () => {
    expect(getQueueQueryScope({ projectId: 'project-1', queueJobsScope: 'active-project' })).toEqual({
      originPrefix: buildProjectQueueItemOriginPrefix('project-1'),
    });

    expect(getQueueQueryScope({ projectId: 'project-1', queueJobsScope: 'all' })).toEqual({});
  });
});
