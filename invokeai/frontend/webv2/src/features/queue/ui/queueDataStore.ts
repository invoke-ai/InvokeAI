import type { QueueItem } from '@features/queue/core/historyTypes';
import type {
  QueueCounts,
  QueueItemDetailReadModel,
  QueueItemReadModel,
  QueueQueryScope,
} from '@features/queue/core/types';

import { getPersonalQueueActivity } from '@features/queue/core/types';
import { invalidateQueueQueries } from '@features/queue/data/queries';
import { getQueueItemDetailOptions, getQueueReadModelOptions } from '@features/queue/publicApi';
import { queryClient } from '@platform/query/client';
import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';

import { getCurrentBatchItems } from './currentBatchItems';
import { clearProgressForQueueItemsWithResults } from './queueProgressCleanup';
import { getQueueOpenWorkElsewhere, useQueueQueryScope } from './queueScope';

const EMPTY_COUNTS: QueueCounts = {
  canceled: 0,
  completed: 0,
  failed: 0,
  inProgress: 0,
  pending: 0,
  queueId: 'default',
  total: 0,
  waiting: 0,
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

/**
 * The viewer's open work in the queue scope beyond `projectItems`, the active project's runs (see
 * `getQueueOpenWorkElsewhere`), so a surface can count the project's runs and this without counting anything twice.
 */
export const useQueueOpenWorkElsewhere = (projectItems: QueueItem[]): number => {
  const model = useQueueReadModel().data;

  return useMemo(() => {
    if (!model) {
      return 0;
    }

    const { inProgress, pending } = getPersonalQueueActivity(model.status.queue);

    return getQueueOpenWorkElsewhere(model, inProgress + pending, { queue: { items: projectItems } });
  }, [model, projectItems]);
};

/** Invalidate queue data; TanStack Query deduplicates concurrent callers. */
export const refreshQueue = (): Promise<void> => invalidateQueueQueries(queryClient);

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

export interface QueueItemDetailState {
  detail: QueueItemDetailReadModel | null;
  error: string | null;
  loadState: 'idle' | 'loading' | 'loaded' | 'error';
  retry: () => void;
  /** A retry of a failed read is in flight; the failure stays shown until it settles. */
  retrying: boolean;
}

/** One item's saved execution data, fetched only while `enabled` (a details dialog is open). */
export const useQueueItemDetail = (itemId: number, enabled: boolean): QueueItemDetailState => {
  const query = useQuery({ ...getQueueItemDetailOptions(itemId), enabled });
  const retry = query.refetch;

  if (!enabled) {
    return { detail: null, error: null, loadState: 'idle', retry, retrying: false };
  }

  if (query.isError) {
    return {
      detail: null,
      // The view words the failure; only a reason the server gave is worth adding to it.
      error: query.error instanceof Error ? query.error.message : null,
      loadState: 'error',
      retry,
      retrying: query.isFetching,
    };
  }

  return {
    detail: query.data ?? null,
    error: null,
    loadState: query.isPending ? 'loading' : 'loaded',
    retry,
    retrying: false,
  };
};
