import { configureIdentityAccountLifecycle } from '@features/identity';
import { queryClient } from '@platform/query/client';
import { accountLifecycle, registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { socketHub } from '@platform/transport/socketHub';

let isConfigured = false;

/** App composes cross-owner account cleanup; lazy features register their caches when loaded. */
export const configureAppAccountLifecycle = (): void => {
  if (isConfigured) {
    return;
  }
  isConfigured = true;

  registerAccountOwnedResource({
    clear: () => {
      // clear() cancels query retryers synchronously before removing caches.
      queryClient.clear();
    },
    name: 'query-client',
  });
  registerAccountOwnedResource({
    clear: () => {
      socketHub.disconnect();
    },
    name: 'socket-hub',
  });
  configureIdentityAccountLifecycle(accountLifecycle);
};
