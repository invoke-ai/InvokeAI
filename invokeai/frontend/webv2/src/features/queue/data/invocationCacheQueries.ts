import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { queryOptions, type QueryClient } from '@tanstack/react-query';

import type { InvocationCacheStatusDTO } from './serverTypes';

import { queueKeys } from './queries';
import { clearInvocationCache, getInvocationCacheStatus, setInvocationCacheEnabled } from './serverApi';

/** The invocation (node output) cache: hit/miss counts, size and whether it is enabled. Admin-only on the server. */
export interface InvocationCacheStatus {
  enabled: boolean;
  hits: number;
  maxSize: number;
  misses: number;
  size: number;
}

const mapInvocationCacheStatus = (dto: InvocationCacheStatusDTO): InvocationCacheStatus => ({
  enabled: dto.enabled,
  hits: dto.hits,
  maxSize: dto.max_size,
  misses: dto.misses,
  size: dto.size,
});

export const invocationCacheKey = [...queueKeys.all, 'invocation-cache'] as const;

/**
 * Statistics have no socket stream. The key sits under the queue's so every queue refresh (status changes, enqueue,
 * cancel) also refetches them, which is when hit and miss counts move; cache commands invalidate them directly.
 */
export const invocationCacheStatusOptions = () =>
  (() => {
    const owner = captureAccountScope();

    return queryOptions({
      queryFn: async ({ signal }): Promise<InvocationCacheStatus> => {
        const status = await getInvocationCacheStatus(AbortSignal.any([signal, owner.signal]));

        assertAccountScopeCurrent(owner);

        return mapInvocationCacheStatus(status);
      },
      queryKey: invocationCacheKey,
      staleTime: 5_000,
    });
  })();

/** Resolves false when the account changed meanwhile: the outcome no longer belongs to anyone who could see it. */
export const runInvocationCacheCommand = async (
  queryClient: QueryClient,
  command: 'clear' | 'disable' | 'enable'
): Promise<boolean> => {
  const owner = captureAccountScope();

  try {
    await (command === 'clear'
      ? clearInvocationCache(owner.signal)
      : setInvocationCacheEnabled(command === 'enable', owner.signal));
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return false;
    }
    throw error;
  }
  if (!isAccountScopeCurrent(owner)) {
    return false;
  }
  await queryClient.invalidateQueries({ queryKey: invocationCacheKey });
  return true;
};
