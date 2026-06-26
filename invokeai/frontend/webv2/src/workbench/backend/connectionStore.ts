import type { BackendConnectionStatus } from '@workbench/types';

import { createExternalStore } from '@workbench/externalStore';

/**
 * Provider-free connection status for the shared backend socket. The socket hub
 * is the sole writer; surfaces that mount no workbench providers (the Launchpad)
 * read it directly, and the editor mirrors it into workbench state via a bridge
 * in `WorkbenchRuntime`. Lives outside the reducer so the connection signal is
 * available everywhere the socket is, not just inside the editor.
 */
export interface ConnectionSnapshot {
  status: BackendConnectionStatus;
  error?: string;
}

const store = createExternalStore<ConnectionSnapshot>({ status: 'connecting' });

export const setConnectionStatus = (status: BackendConnectionStatus, error?: string): void => {
  store.setSnapshot({ error, status });
};

export const getConnectionStatus = (): ConnectionSnapshot => store.getSnapshot();

export const useConnectionStatusSelector = store.useSelector;

export const useConnectionStatus = (): ConnectionSnapshot => store.useSnapshot();

export const subscribeConnection = (listener: () => void): (() => void) => store.subscribe(listener);
