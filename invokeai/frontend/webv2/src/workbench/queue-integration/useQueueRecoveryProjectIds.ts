import type { AccountScope } from '@platform/state/accountLifecycle';

import { queryOptions, useQuery } from '@tanstack/react-query';

import { listQueueRecoveryProjectIds } from './queueRecovery';

export const getQueueRecoveryProjectIdsQueryKey = (owner: AccountScope) =>
  ['workbench', 'queue-recovery-project-ids', owner.accountId, owner.epoch] as const;

export const queueRecoveryProjectIdsQueryOptions = (owner: AccountScope) =>
  queryOptions({
    gcTime: 0,
    queryFn: () => listQueueRecoveryProjectIds(owner),
    queryKey: getQueueRecoveryProjectIdsQueryKey(owner),
    refetchInterval: 10_000,
    retry: false,
    staleTime: 2_000,
  });

export const useQueueRecoveryProjectIds = (owner: AccountScope) => useQuery(queueRecoveryProjectIdsQueryOptions(owner));
