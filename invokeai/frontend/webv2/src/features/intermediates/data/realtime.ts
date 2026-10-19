import type { QueryClient } from '@tanstack/react-query';

import { isOperationSettled, type IntermediatesOperation } from '@features/intermediates/core/types';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { socketHub } from '@platform/transport/socketHub';

import { parseIntermediatesOperationEvent } from './api';
import { intermediatesKeys } from './keys';

const isOperationKey = (queryKey: readonly unknown[], accountKey: readonly unknown[]): boolean =>
  queryKey.length === accountKey.length + 2 &&
  queryKey[accountKey.length] === 'operation' &&
  accountKey.every((segment, index) =>
    typeof segment === 'object' && segment !== null
      ? JSON.stringify(segment) === JSON.stringify(queryKey[index])
      : segment === queryKey[index]
  );

/**
 * Keeps the open manager current. Socket events replace the operation record; whichever path delivers a settled
 * operation first (socket or the polled query) invalidates the account's counts exactly once, since every row the
 * operation touched changed. Admins receive other accounts' events too, which is exactly when their totals change.
 * A reconnect invalidates everything, since events during the gap were lost.
 */
export const attachIntermediatesRealtime = (queryClient: QueryClient): (() => void) => {
  const owner = captureAccountScope();
  const accountKey = intermediatesKeys.forAccount(owner);
  const settledOperationIds = new Set<string>();

  const noteSettled = (operation: IntermediatesOperation) => {
    if (!isOperationSettled(operation) || settledOperationIds.has(operation.operationId)) {
      return;
    }
    settledOperationIds.add(operation.operationId);
    void queryClient.invalidateQueries({ queryKey: intermediatesKeys.summaries(owner) });
  };

  const detachCache = queryClient.getQueryCache().subscribe((event) => {
    if (event.type !== 'updated' || event.action.type !== 'success' || !isAccountScopeCurrent(owner)) {
      return;
    }
    if (isOperationKey(event.query.queryKey, accountKey)) {
      noteSettled(event.action.data as IntermediatesOperation);
    }
  });

  const detachOperation = socketHub.on('intermediates_operation_changed', (payload: never) => {
    const operation: IntermediatesOperation | null = parseIntermediatesOperationEvent(payload);
    if (!operation || !isAccountScopeCurrent(owner)) {
      return;
    }

    queryClient.setQueryData(intermediatesKeys.operation(owner, operation.operationId), operation);
    noteSettled(operation);
  });

  // The hub reports the current status on subscribe; only a later transition to connected is a reconnect.
  let wasConnected: boolean | null = null;
  const detachConnection = socketHub.onConnectionChange((status) => {
    const isConnected = status === 'connected';

    if (isConnected && wasConnected === false && isAccountScopeCurrent(owner)) {
      void queryClient.invalidateQueries({ queryKey: accountKey });
    }
    wasConnected = isConnected;
  });

  return () => {
    detachCache();
    detachOperation();
    detachConnection();
  };
};
