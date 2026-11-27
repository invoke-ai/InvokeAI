import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { InvocationSourceId } from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';

import { normalizeGenerateWidgetValues } from '@features/generation/settings';
import { parseQueueItemOrigin, parseQueueItemOriginProjectId } from '@features/queue/contracts';
import { normalizeVideoWidgetValues, type VideoWidgetValues } from '@features/video';
import { assertAccountScopeCurrent, captureAccountScope, type AccountScope } from '@platform/state/accountLifecycle';
import { type QueryClient, useQuery } from '@tanstack/react-query';
import { isRecord } from '@workbench/canvasMigration';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';

import { createAccountOwnedQueueRecallCache } from './queueRecallCache';

export interface LocalRecallSnapshot {
  canvasMetadata?: Record<string, unknown>;
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

/** This browser's durable snapshot of a queue item another tab, or an earlier page, submitted. */
const durableRecallSnapshotOptions = (owner: AccountScope, projectId: string | null, queueItemId: string | null) => ({
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
        ...(isRecord(snapshot.canvasMetadata) ? { canvasMetadata: snapshot.canvasMetadata } : {}),
        generateValues: snapshot.generateValues
          ? (normalizeGenerateWidgetValues(snapshot.generateValues) ?? undefined)
          : undefined,
        videoValues: snapshot.videoValues ? (normalizeVideoWidgetValues(snapshot.videoValues) ?? undefined) : undefined,
      };
    } finally {
      cache.close();
    }
  },
});

const toLocalRecallSnapshot = (item: Pick<Project, 'queue'>['queue']['items'][number]): LocalRecallSnapshot => ({
  ...item.snapshot.recall,
  sourceId: item.snapshot.sourceId,
});

/** `undefined` means the durable lookup is pending; `null` means it completed without a usable snapshot. */
export const useLocalRecallSnapshot = (origin?: string | null): LocalRecallSnapshot | null | undefined => {
  const owner = captureAccountScope();
  const localItem = useWorkbenchSelector((state) => findLocalQueueItem(state.projects, origin));
  const queueItemId = parseQueueItemOrigin(origin);
  const projectId = parseQueueItemOriginProjectId(origin);
  const { data, isPending } = useQuery({
    ...durableRecallSnapshotOptions(owner, projectId, queueItemId),
    enabled: !localItem && queueItemId !== null,
  });
  if (localItem) {
    return toLocalRecallSnapshot(localItem);
  }
  if (queueItemId !== null && isPending) {
    return undefined;
  }
  return data ?? null;
};

/** The same answer as `useLocalRecallSnapshot`, read once for a caller outside React; null when there is none. */
export const loadLocalRecallSnapshot = async ({
  origin,
  projects,
  queryClient,
}: {
  origin?: string | null;
  projects: readonly Pick<Project, 'id' | 'queue'>[];
  queryClient: QueryClient;
}): Promise<LocalRecallSnapshot | null> => {
  const localItem = findLocalQueueItem(projects, origin);
  if (localItem) {
    return toLocalRecallSnapshot(localItem);
  }
  const queueItemId = parseQueueItemOrigin(origin);
  if (queueItemId === null) {
    return null;
  }
  try {
    return await queryClient.fetchQuery(
      durableRecallSnapshotOptions(captureAccountScope(), parseQueueItemOriginProjectId(origin), queueItemId)
    );
  } catch {
    return null;
  }
};
