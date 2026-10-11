import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
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

/** Statistics have no socket stream; they refresh on mount and after each cache command. */
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

export const runInvocationCacheCommand = async (
  queryClient: QueryClient,
  command: 'clear' | 'disable' | 'enable'
): Promise<void> => {
  const owner = captureAccountScope();

  await (command === 'clear' ? clearInvocationCache() : setInvocationCacheEnabled(command === 'enable'));
  assertAccountScopeCurrent(owner);
  await queryClient.invalidateQueries({ queryKey: invocationCacheKey });
};
