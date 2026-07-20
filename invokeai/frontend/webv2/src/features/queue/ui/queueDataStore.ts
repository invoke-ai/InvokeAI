import type { QueueCounts, QueueItemReadModel, QueueQueryScope } from '@features/queue/core/types';

import { getQueueReadModelOptions } from '@features/queue/publicApi';
import { queryClient } from '@platform/query/client';
import { useQuery } from '@tanstack/react-query';

import { getCurrentBatchItems } from './currentBatchItems';
import { clearProgressForQueueItemsWithResults } from './queueProgressCleanup';
import { useQueueQueryScope } from './queueScope';

const EMPTY_COUNTS: QueueCounts = {
  canceled: 0,
  completed: 0,
  failed: 0,
  inProgress: 0,
  pending: 0,
  queueId: 'default',
  total: 0,
};

const onQueueRead = ({
  current,
  items,
  next,
}: {
  current: QueueItemReadModel | null;
  items: QueueItemReadModel[];
  next: QueueItemReadModel | null;
}): void => {
  clearProgressForQueueItemsWithResults([current, next, ...items].filter((item) => item !== null));
};

const useQueueReadModel = () => {
  const scope = useQueueQueryScope();

  return useQuery(getQueueReadModelOptions(scope, onQueueRead));
};

/** Invalidate queue data; TanStack Query deduplicates concurrent callers. */
export const refreshQueue = async (): Promise<void> => {
  await queryClient.invalidateQueries({ queryKey: ['queue'] });
};

/** Imperative prefetch for non-React composition roots. */
export const ensureQueueLoaded = (scope: QueueQueryScope = {}): void => {
  void queryClient.ensureQueryData(getQueueReadModelOptions(scope, onQueueRead));
};

export const useQueueCounts = (): QueueCounts => useQueueReadModel().data?.status.queue ?? EMPTY_COUNTS;

/** The processor is "paused" when it has been stopped. */
export const useIsProcessorPaused = (): boolean => useQueueReadModel().data?.status.processor.isStarted === false;

export const useNowNextItems = (): { current: QueueItemReadModel | null; next: QueueItemReadModel | null } => {
  const data = useQueueReadModel().data;

  return { current: data?.current ?? null, next: data?.next ?? null };
};

export const useCurrentBatchItems = (): QueueItemReadModel[] => {
  const data = useQueueReadModel().data;

  return data ? getCurrentBatchItems(data) : [];
};

export const useRecentItems = (): QueueItemReadModel[] => useQueueReadModel().data?.items ?? [];

export const useQueueLoadState = (): { loadState: 'loading' | 'loaded' | 'error'; error: string | null } => {
  const query = useQueueReadModel();

  if (query.isError) {
    return { error: query.error instanceof Error ? query.error.message : 'Failed to load queue', loadState: 'error' };
  }

  return { error: null, loadState: query.isPending ? 'loading' : 'loaded' };
};
