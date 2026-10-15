import type { IntermediatesOperation, IntermediatesSummary } from '@features/intermediates/core/types';

import { isOperationSettled } from '@features/intermediates/core/types';
import { assertAccountScopeCurrent, captureAccountScope, type AccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { queryOptions } from '@tanstack/react-query';

import type { IntermediatesSummaryParams } from './keys';

import { getIntermediatesOperation, getIntermediatesSummary } from './api';
import { intermediatesKeys } from './keys';

export const INTERMEDIATES_PAGE_SIZE = 50;

const fenced = <T>(owner: AccountScope, request: (signal: AbortSignal) => Promise<T>, signal: AbortSignal) =>
  request(AbortSignal.any([signal, owner.signal])).then((result) => {
    assertAccountScopeCurrent(owner);
    return result;
  });

export const intermediatesSummaryQueryOptions = (params: IntermediatesSummaryParams, owner = captureAccountScope()) =>
  queryOptions<IntermediatesSummary>({
    queryFn: ({ signal }) => fenced(owner, (fencedSignal) => getIntermediatesSummary(params, fencedSignal), signal),
    queryKey: intermediatesKeys.summary(owner, params),
    refetchInterval: (query) => (query.state.data?.measuring ? 5_000 : false),
    staleTime: 15_000,
  });

/** Polls while the operation runs as a fallback for a missed socket event; stops once it settles. */
export const intermediatesOperationQueryOptions = (operationId: string, owner = captureAccountScope()) =>
  queryOptions<IntermediatesOperation>({
    queryFn: ({ signal }) =>
      fenced(owner, (fencedSignal) => getIntermediatesOperation(operationId, fencedSignal), signal),
    queryKey: intermediatesKeys.operation(owner, operationId),
    retry: (failureCount, error) => !(error instanceof ApiError && error.status === 404) && failureCount < 1,
    refetchInterval: (query) =>
      (query.state.error instanceof ApiError && query.state.error.status === 404) ||
      (query.state.data && isOperationSettled(query.state.data))
        ? false
        : 2_000,
    staleTime: 1_000,
  });
