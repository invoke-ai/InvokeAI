import type { IntermediatesSummarySort } from '@features/intermediates/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';

export interface IntermediatesSummaryParams {
  ownerId?: string | null;
  projectId?: string | null;
  search?: string;
  sort?: IntermediatesSummarySort;
  order?: 'asc' | 'desc';
  offset?: number;
  limit?: number;
}

const getAccountKey = (owner: AccountScope) => ({ accountId: owner.accountId, epoch: owner.epoch });

/** Keys carry the account so a sign-out never serves one account's counts to the next. */
export const intermediatesKeys = {
  all: ['intermediates'] as const,
  forAccount: (owner: AccountScope) => [...intermediatesKeys.all, getAccountKey(owner)] as const,
  summary: (owner: AccountScope, params: IntermediatesSummaryParams) =>
    [...intermediatesKeys.forAccount(owner), 'summary', params] as const,
  summaries: (owner: AccountScope) => [...intermediatesKeys.forAccount(owner), 'summary'] as const,
  operation: (owner: AccountScope, operationId: string) =>
    [...intermediatesKeys.forAccount(owner), 'operation', operationId] as const,
};
