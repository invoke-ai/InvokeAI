import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { InvocationSourceId } from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';

import { normalizeGenerateWidgetValues } from '@features/generation/settings';
import { parseQueueItemOrigin, parseQueueItemOriginProjectId } from '@features/queue/contracts';
import { normalizeVideoWidgetValues, type VideoWidgetValues } from '@features/video';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { useQuery } from '@tanstack/react-query';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';

import { createAccountOwnedQueueRecallCache } from './queueRecallCache';

export interface LocalRecallSnapshot {
  generateValues?: GenerateWidgetValues;
  sourceId: InvocationSourceId;
  videoValues?: VideoWidgetValues;
}

const findLocalQueueItem = (projects: readonly Pick<Project, 'id' | 'queue'>[], origin?: string | null) => {
  const localId = parseQueueItemOrigin(origin);
  const projectId = parseQueueItemOriginProjectId(origin);
  if (!localId) {
    return null;
  }
  for (const project of projects) {
    if (projectId && project.id !== projectId) {
      continue;
    }
    const item = project.queue.items.find((candidate) => candidate.id === localId);
    if (item) {
      return item;
    }
  }
  return null;
};

export const getLocalQueueItemSource = (
  projects: readonly Pick<Project, 'id' | 'queue'>[],
  origin?: string | null
): InvocationSourceId | null => findLocalQueueItem(projects, origin)?.snapshot.sourceId ?? null;

/** `undefined` means the durable lookup is pending; `null` means it completed without a usable snapshot. */
export const useLocalRecallSnapshot = (origin?: string | null): LocalRecallSnapshot | null | undefined => {
  const owner = captureAccountScope();
  const localItem = useWorkbenchSelector((state) => findLocalQueueItem(state.projects, origin));
  const queueItemId = parseQueueItemOrigin(origin);
  const projectId = parseQueueItemOriginProjectId(origin);
  const { data, isPending } = useQuery({
    enabled: !localItem && queueItemId !== null,
    queryKey: ['queue-recall', owner.epoch, projectId, queueItemId],
    retry: false,
    staleTime: 30_000,
    async queryFn(): Promise<LocalRecallSnapshot | null> {
      if (!queueItemId) {
        return null;
      }
      const cache = await createAccountOwnedQueueRecallCache(owner);
      try {
        const result = await cache.get(queueItemId);
        assertAccountScopeCurrent(owner);
        if (result.kind !== 'found' || (projectId && result.snapshot.projectId !== projectId)) {
          return null;
        }
        const snapshot = result.snapshot;
        return {
          sourceId: snapshot.sourceId,
          generateValues: snapshot.generateValues
            ? (normalizeGenerateWidgetValues(snapshot.generateValues) ?? undefined)
            : undefined,
          videoValues: snapshot.videoValues
            ? (normalizeVideoWidgetValues(snapshot.videoValues) ?? undefined)
            : undefined,
        };
      } finally {
        cache.close();
      }
    },
  });
  if (localItem) {
    return { ...localItem.snapshot.recall, sourceId: localItem.snapshot.sourceId };
  }
  if (queueItemId !== null && isPending) {
    return undefined;
  }
  return data ?? null;
};
