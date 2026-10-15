import type { InfiniteData } from '@tanstack/react-query';

import {
  getLibraryWorkflowRecord,
  listLibraryWorkflows,
  type ListWorkflowsParams,
  type WorkflowLibraryPage,
  type WorkflowRecordDTO,
} from './api';

export const savedWorkflowDetailQueryKey = (workflowId: string) =>
  ['workflow', 'call-saved', 'detail', workflowId] as const;

export const savedWorkflowPickerQueryKeyPrefix = ['workflow', 'call-saved', 'picker'] as const;

export const isSavedWorkflowDetailQueryKey = (
  queryKey: readonly unknown[]
): queryKey is readonly ['workflow', 'call-saved', 'detail', string] =>
  queryKey[0] === 'workflow' &&
  queryKey[1] === 'call-saved' &&
  queryKey[2] === 'detail' &&
  typeof queryKey[3] === 'string' &&
  queryKey[3].length > 0;

type SavedWorkflowDetailQueryLike = {
  state: {
    data?: unknown;
    fetchStatus: 'fetching' | 'paused' | 'idle';
    isInvalidated: boolean;
    status: 'pending' | 'error' | 'success';
  };
};

export interface SavedWorkflowDetailFetchOptions {
  retryErrors?: boolean;
}

/** Authorizes one recovery fetch after an unrequested stale-detail revalidation failure. */
export const shouldRetrySavedWorkflowDetailAfterFailure = (
  retryWasAuthorized: boolean,
  hasExistingDetail: boolean
): boolean => !retryWasAuthorized && hasExistingDetail;

export const shouldFetchSavedWorkflowDetail = (
  query: SavedWorkflowDetailQueryLike | undefined,
  options: SavedWorkflowDetailFetchOptions = {}
): boolean =>
  query === undefined ||
  (query.state.fetchStatus === 'idle' &&
    ((query.state.status === 'pending' && query.state.data === undefined) ||
      (query.state.isInvalidated && query.state.status === 'success') ||
      (options.retryErrors === true && query.state.status === 'error')));

export const getSavedWorkflowDetailQueryStatus = (
  query: SavedWorkflowDetailQueryLike | undefined
): 'missing' | 'loading' | 'ready' | 'error' => {
  if (!query) {
    return 'missing';
  }

  if (query.state.status === 'error') {
    return 'error';
  }

  if (query.state.data !== undefined && query.state.data !== null) {
    return 'ready';
  }

  if (query.state.status !== 'success' || query.state.fetchStatus !== 'idle') {
    return 'loading';
  }

  return query.state.data !== undefined && query.state.data !== null ? 'ready' : 'error';
};

export const savedWorkflowDetailQueryOptions = (workflowId: string) => ({
  queryKey: savedWorkflowDetailQueryKey(workflowId),
  queryFn: ({ signal }: { signal: AbortSignal }): Promise<WorkflowRecordDTO> =>
    getLibraryWorkflowRecord(workflowId, signal),
  gcTime: Infinity,
  retry: false,
  staleTime: 30_000,
});

export const savedWorkflowPickerQueryOptions = (params: ListWorkflowsParams) => ({
  queryKey: [...savedWorkflowPickerQueryKeyPrefix, params] as const,
  queryFn: ({ pageParam, signal }: { pageParam: number; signal: AbortSignal }): Promise<WorkflowLibraryPage> =>
    listLibraryWorkflows({ ...params, page: pageParam, signal }),
  staleTime: 30_000,
  initialPageParam: 0,
  getNextPageParam: (lastPage: WorkflowLibraryPage): number | undefined =>
    lastPage.page + 1 < lastPage.pages ? lastPage.page + 1 : undefined,
});

export const getWorkflowPagesItems = (
  data: InfiniteData<WorkflowLibraryPage, unknown> | undefined
): WorkflowLibraryPage['items'] => data?.pages.flatMap((page) => page.items) ?? [];
