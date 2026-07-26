import { configureIdentityAccountLifecycle } from '@features/identity';
import { queryClient } from '@platform/query/client';
import { accountLifecycle, registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { socketHub } from '@platform/transport/socketHub';

let isConfigured = false;

/**
 * App composition for resources whose implementations Identity must not know.
 * Lazy feature modules register their own account-owned caches with the same
 * lifecycle when they load.
 */
export const configureAppAccountLifecycle = (): void => {
  if (isConfigured) {
    return;
  }
  isConfigured = true;

  registerAccountOwnedResource({
    clear: () => {
      // QueryClient.clear() synchronously destroys queries, which cancels their
      // retryers before removing both query and mutation caches.
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
