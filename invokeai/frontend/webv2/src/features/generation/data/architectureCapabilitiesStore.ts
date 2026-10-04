import {
  getArchitectureCapabilitiesRevision,
  onArchitectureCapabilitiesChanged,
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createTrailingSingleFlight } from '@platform/state/singleFlight';
import { getApiErrorMessage } from '@platform/transport/http';

import { getArchitectureCapabilities } from './architectureCapabilitiesApi';

/** Fetch status is separate from authoritative core rows; the table is static per backend build. */

export interface ArchitectureCapabilitiesSnapshot {
  status: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
  /** Revision zero means absent; observe registry identity rather than request status. */
  revision: number;
}

const EMPTY_SNAPSHOT: ArchitectureCapabilitiesSnapshot = { error: null, revision: 0, status: 'idle' };
const store = createExternalStore<ArchitectureCapabilitiesSnapshot>(EMPTY_SNAPSHOT);

// Publish from registry changes so direct seeds and clears also notify subscribers.
onArchitectureCapabilitiesChanged(() => store.patchSnapshot({ revision: getArchitectureCapabilitiesRevision() }));

const refreshFlight = createTrailingSingleFlight();

/** Whether anything in this session has asked for the table; see the account-change re-arm below. */
let isRequested = false;

registerAccountOwnedResource({
  clear: () => {
    refreshFlight.reset();
    // Clear the registry before publishing revision zero to prevent old-account reads.
    resetArchitectureCapabilities();
    store.setSnapshot(EMPTY_SNAPSHOT);

    // Refetch after account rotation; the one-time boot mount cannot replace aborted work.
    if (isRequested && captureAccountScope().accountId !== null) {
      void refreshArchitectureCapabilities();
    }
  },
  name: 'architecture-capabilities',
});

export const refreshArchitectureCapabilities = (): Promise<void> =>
  refreshFlight.run(() => {
    const owner = captureAccountScope();
    store.patchSnapshot({ status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'loading' });

    return getArchitectureCapabilities(owner.signal)
      .then((rows) => {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        setArchitectureCapabilities(rows);
        store.patchSnapshot({ error: null, status: 'loaded' });
      })
      .catch((error: unknown) => {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        store.patchSnapshot({
          error: getApiErrorMessage(error, 'Failed to load model capabilities.'),
          status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'error',
        });
      });
  });

/** This promise never rejects; inspect snapshot status after the joined or started load settles. */
export const ensureArchitectureCapabilitiesLoaded = (): Promise<void> => {
  isRequested = true;

  const { status } = store.getSnapshot();

  if (status === 'idle' || status === 'error') {
    return refreshArchitectureCapabilities();
  }

  return refreshFlight.inflight() ?? Promise.resolve();
};

export const getArchitectureCapabilitiesSnapshot = (): ArchitectureCapabilitiesSnapshot => store.getSnapshot();

export const subscribeArchitectureCapabilities = store.subscribe;

export const useArchitectureCapabilitiesSelector = store.useSelector;
