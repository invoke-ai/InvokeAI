import type { WorkingMemorySummary } from '@features/models/core/workingMemory';

import { assertAccountScopeCurrent, captureAccountScope, type AccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { queryOptions } from '@tanstack/react-query';

import { getWorkingMemoryPlan, getWorkingMemoryStatus, getWorkingMemorySummary } from './workingMemoryApi';

/** Polls while a run is pending or running and the socket is down, which includes a backend without one (the mock). */
export const WORKING_MEMORY_POLL_MS = 2_000;
/** With a socket, its events drive updates; this only catches a missed one. */
export const WORKING_MEMORY_FALLBACK_POLL_MS = 30_000;

const accountKey = (owner: AccountScope) => ({ accountId: owner.accountId, epoch: owner.epoch });

/** Keys carry the account so a sign-out never serves one account's calibration to the next. */
export const workingMemoryKeys = {
  forAccount: (owner: AccountScope) => ['models', 'workingMemory', accountKey(owner)] as const,
  plan: (owner: AccountScope, modelKey: string) => [...workingMemoryKeys.plans(owner), modelKey] as const,
  plans: (owner: AccountScope) => [...workingMemoryKeys.forAccount(owner), 'plan'] as const,
  status: (owner: AccountScope) => [...workingMemoryKeys.forAccount(owner), 'status'] as const,
  summaries: (owner: AccountScope) => [...workingMemoryKeys.forAccount(owner), 'summary'] as const,
  summary: (owner: AccountScope, modelKey: string) => [...workingMemoryKeys.summaries(owner), modelKey] as const,
};

const fenced = <T>(owner: AccountScope, request: (signal: AbortSignal) => Promise<T>, signal: AbortSignal) =>
  request(AbortSignal.any([signal, owner.signal])).then((result) => {
    assertAccountScopeCurrent(owner);
    return result;
  });

export const hasRunningRun = (summary: WorkingMemorySummary | null | undefined): boolean =>
  summary?.runs.some((run) => run.status === 'running') ?? false;

/**
 * The plan sizes its cases for the GPU memory free on the server when it is read and for the measurements stored by
 * then. It is read again whenever the section opens and right before a calibration starts
 * (`refreshWorkingMemoryPlan`, `fetchCurrentWorkingMemoryPlan`); in between it holds.
 */
export const workingMemoryPlanQueryOptions = (modelKey: string, owner = captureAccountScope()) =>
  queryOptions({
    queryFn: ({ signal }) => fenced(owner, (fencedSignal) => getWorkingMemoryPlan(modelKey, fencedSignal), signal),
    queryKey: workingMemoryKeys.plan(owner, modelKey),
    staleTime: 60_000,
  });

/** Null when the server has no current device class to summarise; the status then says why. */
export const workingMemorySummaryQueryOptions = (
  modelKey: string,
  { isConnected, isSubmitting }: { isConnected: boolean; isSubmitting: boolean },
  owner = captureAccountScope()
) =>
  queryOptions({
    queryFn: ({ signal }) =>
      fenced(
        owner,
        (fencedSignal) =>
          getWorkingMemorySummary(modelKey, fencedSignal).catch((error: unknown) => {
            if (error instanceof ApiError && error.status === 409) {
              return null;
            }
            throw error;
          }),
        signal
      ),
    queryKey: workingMemoryKeys.summary(owner, modelKey),
    refetchInterval: (query) =>
      isSubmitting || hasRunningRun(query.state.data)
        ? isConnected
          ? WORKING_MEMORY_FALLBACK_POLL_MS
          : WORKING_MEMORY_POLL_MS
        : false,
    staleTime: 5_000,
  });

export const workingMemoryStatusQueryOptions = (owner = captureAccountScope()) =>
  queryOptions({
    queryFn: ({ signal }) => fenced(owner, (fencedSignal) => getWorkingMemoryStatus(fencedSignal), signal),
    queryKey: workingMemoryKeys.status(owner),
    staleTime: 15_000,
  });
