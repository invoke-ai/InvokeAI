import { QueryClient } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  getSavedWorkflowDetailQueryStatus,
  isSavedWorkflowDetailQueryKey,
  savedWorkflowDetailQueryOptions,
  shouldRetrySavedWorkflowDetailAfterFailure,
  savedWorkflowPickerQueryOptions,
  shouldFetchSavedWorkflowDetail,
} from './savedWorkflowQueries';

describe('saved workflow detail query policy', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('recognizes only detail query keys with a workflow id', () => {
    expect(isSavedWorkflowDetailQueryKey(['workflow', 'call-saved', 'detail', 'workflow-1'])).toBe(true);
    expect(isSavedWorkflowDetailQueryKey(['workflow', 'call-saved', 'picker', 'workflow-1'])).toBe(false);
    expect(isSavedWorkflowDetailQueryKey(['workflow', 'call-saved', 'detail'])).toBe(false);
  });

  it('fetches missing or invalidated detail queries and only retries explicitly invalidated errors', () => {
    expect(shouldFetchSavedWorkflowDetail(undefined)).toBe(true);
    expect(
      shouldFetchSavedWorkflowDetail({ state: { fetchStatus: 'idle', isInvalidated: true, status: 'success' } })
    ).toBe(true);
    expect(
      shouldFetchSavedWorkflowDetail({ state: { fetchStatus: 'idle', isInvalidated: false, status: 'success' } })
    ).toBe(false);
    expect(
      shouldFetchSavedWorkflowDetail({ state: { fetchStatus: 'fetching', isInvalidated: true, status: 'pending' } })
    ).toBe(false);
    expect(
      shouldFetchSavedWorkflowDetail({ state: { fetchStatus: 'idle', isInvalidated: true, status: 'error' } })
    ).toBe(false);
    expect(
      shouldFetchSavedWorkflowDetail(
        { state: { fetchStatus: 'idle', isInvalidated: true, status: 'error' } },
        { retryErrors: true }
      )
    ).toBe(true);
    expect(
      shouldFetchSavedWorkflowDetail(
        { state: { fetchStatus: 'idle', isInvalidated: false, status: 'error' } },
        {
          retryErrors: true,
        }
      )
    ).toBe(true);
  });

  it('classifies detail query state for reconciliation', () => {
    expect(getSavedWorkflowDetailQueryStatus(undefined)).toBe('missing');
    expect(
      getSavedWorkflowDetailQueryStatus({ state: { fetchStatus: 'fetching', isInvalidated: false, status: 'pending' } })
    ).toBe('loading');
    expect(
      getSavedWorkflowDetailQueryStatus({ state: { fetchStatus: 'idle', isInvalidated: false, status: 'error' } })
    ).toBe('error');
    expect(
      getSavedWorkflowDetailQueryStatus({
        state: { data: { workflow_id: 'workflow-1' }, fetchStatus: 'idle', isInvalidated: false, status: 'success' },
      })
    ).toBe('ready');
    expect(
      getSavedWorkflowDetailQueryStatus({
        state: { data: { workflow_id: 'workflow-1' }, fetchStatus: 'fetching', isInvalidated: true, status: 'success' },
      })
    ).toBe('ready');
    expect(
      getSavedWorkflowDetailQueryStatus({
        state: { data: { workflow_id: 'workflow-1' }, fetchStatus: 'idle', isInvalidated: true, status: 'error' },
      })
    ).toBe('error');
  });

  it('does not retry a failed detail lookup', () => {
    expect(savedWorkflowDetailQueryOptions('workflow-1').retry).toBe(false);
    expect(savedWorkflowDetailQueryOptions('workflow-1').gcTime).toBe(Infinity);
    expect(savedWorkflowPickerQueryOptions({ isPublic: true, page: 0 }).staleTime).toBe(30_000);
  });

  it('allows one stale-detail recovery retry without retrying indefinitely', () => {
    expect(shouldRetrySavedWorkflowDetailAfterFailure(false, true)).toBe(true);
    expect(shouldRetrySavedWorkflowDetailAfterFailure(true, true)).toBe(false);
    expect(shouldRetrySavedWorkflowDetailAfterFailure(false, false)).toBe(false);
  });

  it('recovers an invalidated failed lookup with one authorized retry', async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const detailOptions = savedWorkflowDetailQueryOptions('workflow-1');
    let attempts = 0;
    const options = {
      ...detailOptions,
      queryFn: () => {
        attempts += 1;

        if (attempts === 1) {
          throw new Error('temporary failure');
        }

        return { workflow_id: 'workflow-1' } as never;
      },
    };

    try {
      await expect(queryClient.fetchQuery(options)).rejects.toThrow('temporary failure');
      const failedQuery = queryClient.getQueryCache().find({ queryKey: detailOptions.queryKey });

      expect(shouldFetchSavedWorkflowDetail(failedQuery)).toBe(false);

      await queryClient.invalidateQueries({ exact: true, queryKey: detailOptions.queryKey, refetchType: 'none' });
      const invalidatedQuery = queryClient.getQueryCache().find({ queryKey: detailOptions.queryKey });

      expect(shouldFetchSavedWorkflowDetail(invalidatedQuery, { retryErrors: true })).toBe(true);
      await queryClient.ensureQueryData({ ...options, revalidateIfStale: true });

      expect(attempts).toBe(2);
      expect(
        getSavedWorkflowDetailQueryStatus(queryClient.getQueryCache().find({ queryKey: detailOptions.queryKey }))
      ).toBe('ready');
    } finally {
      queryClient.clear();
    }
  });

  it('retains an observerless selected detail beyond the default cache lifetime', async () => {
    vi.useFakeTimers();
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const detailOptions = savedWorkflowDetailQueryOptions('workflow-1');

    try {
      await queryClient.fetchQuery({
        ...detailOptions,
        queryFn: () => Promise.resolve({ workflow_id: 'workflow-1' }) as never,
      });
      vi.advanceTimersByTime(5 * 60_000 + 1);

      expect(queryClient.getQueryCache().find({ queryKey: detailOptions.queryKey })).toBeDefined();
    } finally {
      queryClient.clear();
    }
  });
});

describe('detail queries built without a fetch', () => {
  // A pending/idle detail query may have no active fetch and infinite lifetime; selecting a cached list entry must
  // start its detail request.
  it('fetches a query that was built but never fetched', () => {
    expect(
      shouldFetchSavedWorkflowDetail({ state: { fetchStatus: 'idle', isInvalidated: false, status: 'pending' } })
    ).toBe(true);
  });
});
