import type { QueueItem } from '@features/queue/core/historyTypes';
import type {
  QueueCounts,
  QueueItemDetailReadModel,
  QueueItemReadModel,
  QueueQueryScope,
  QueueReadModel,
  QueueStatusReadModel,
} from '@features/queue/core/types';
import type { QueryClient, QueryFunction } from '@tanstack/react-query';

import { getPersonalQueueActivity } from '@features/queue/core/types';
import { queueBackend } from '@features/queue/data/httpRealtimeQueueBackend';
import { invalidateQueueQueries, queueStatusOptions } from '@features/queue/data/queries';
import { getQueueItemDetailOptions, getQueueReadModelOptions } from '@features/queue/publicApi';
import { queryClient } from '@platform/query/client';
import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';

import { getCurrentBatchItems } from './currentBatchItems';
import { clearProgressForQueueItemsWithResults } from './queueProgressCleanup';
import { getQueueOpenWorkElsewhere, useQueueQueryScope } from './queueScope';

const WHOLE_QUEUE_SCOPE: QueueQueryScope = {};

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

/**
 * Runs a query's read now and stores what it finds. A read already in flight is neither joined (it may have started
 * before what the caller just learned) nor cancelled (others, such as the command palette's queue search, may be
 * awaiting it); if it lands later with an older reading, the next invalidation's read reconciles it.
 */
const readNow = async <T, Key extends readonly unknown[]>(
  client: QueryClient,
  options: { queryFn?: unknown; queryKey: Key },
  signal: AbortSignal
): Promise<T> => {
  const queryFn = options.queryFn as QueryFunction<T, Key>;
  const data = await queryFn({ client, meta: undefined, queryKey: options.queryKey, signal });

  client.setQueryData<T>(options.queryKey as readonly unknown[], data);
  return data;
};

/** The scope's read model as the server has it now, read after the call. Null when the server cannot say. */
export const readQueueReadModel = (
  client: QueryClient,
  scope: QueueQueryScope,
  signal: AbortSignal
): Promise<QueueReadModel | null> => readNow(client, getQueueReadModelOptions(scope, onQueueRead), signal);

/** The whole queue's status as the server has it now, read after the call, under the key the Launchpad shares. */
export const readWholeQueueStatus = (client: QueryClient, signal: AbortSignal): Promise<QueueStatusReadModel> =>
  readNow(client, queueStatusOptions(queueBackend, WHOLE_QUEUE_SCOPE), signal);

/**
 * The whole queue's status, read only while the view is scoped: an unscoped read model's counts are already the whole
 * queue's. Shares its key with the Launchpad's status band, so it is not a request of its own per view.
 */
export const useWholeQueueStatus = (scope: QueueQueryScope): QueueStatusReadModel | null =>
  useQuery({ ...queueStatusOptions(queueBackend, WHOLE_QUEUE_SCOPE), enabled: Boolean(scope.originPrefix) }).data ??
  null;

/** The scope's loaded read model, for the actions that decide from its whole reading. */
export const useQueueReadModelData = (): QueueReadModel | null => useQueueReadModel().data ?? null;

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
