import { createExternalStore } from '@platform/state/externalStore';

import type { BackendConnectionStatus } from './types';

/** The socket hub alone writes this provider-free store; Launchpad reads directly and Workbench mirrors it. */
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
