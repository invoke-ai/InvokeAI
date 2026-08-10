import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createTrailingSingleFlight } from '@platform/state/singleFlight';

import {
  getExternalProviderConfigs,
  resetExternalProviderConfig,
  setExternalProviderConfig,
  type ExternalProviderConfig,
} from './api';

/**
 * Shared cache of external image-provider configs. One store keeps every
 * consumer consistent: saving a key on the Keys tab immediately flips the
 * Configure/Installed affordances in the add-models starter list, with no
 * remount or refetch.
 */

export interface ExternalProvidersSnapshot {
  /** null = not fetched yet. */
  configs: ExternalProviderConfig[] | null;
  status: 'idle' | 'loading' | 'loaded' | 'error';
  /** Failure message when one exists; `status` is the failure signal. The UI supplies the translated fallback. */
  error: string | null;
}

const EMPTY_EXTERNAL_PROVIDERS_SNAPSHOT: ExternalProvidersSnapshot = { configs: null, error: null, status: 'idle' };
const store = createExternalStore<ExternalProvidersSnapshot>(EMPTY_EXTERNAL_PROVIDERS_SNAPSHOT);

const refreshFlight = createTrailingSingleFlight();

registerAccountOwnedResource({
  clear: () => {
    refreshFlight.reset();
    store.setSnapshot(EMPTY_EXTERNAL_PROVIDERS_SNAPSHOT);
  },
  name: 'external-providers',
});

/** Re-fetch the configs; rejects on failure (after recording it) so callers can toast. */
export const refreshExternalProviders = (): Promise<void> =>
  refreshFlight.run(() => {
    const owner = captureAccountScope();

    store.patchSnapshot({ status: store.getSnapshot().configs ? 'loaded' : 'loading' });

    return getExternalProviderConfigs(owner.signal)
      .then((configs) => {
        if (isAccountScopeCurrent(owner)) {
          store.patchSnapshot({ configs, error: null, status: 'loaded' });
        }
      })
      .catch((error: unknown) => {
        if (isAccountScopeCurrent(owner)) {
          store.patchSnapshot({
            error: error instanceof Error ? error.message : null,
            status: store.getSnapshot().configs ? 'loaded' : 'error',
          });
        }
        throw error;
      });
  });

/** Fetch on first use; callers share the request and can catch its failure. */
export const ensureExternalProvidersLoaded = (): Promise<void> => {
  const { status } = store.getSnapshot();

  if (status === 'idle' || status === 'error') {
    return refreshExternalProviders();
  }

  return refreshFlight.inflight() ?? Promise.resolve();
};

const replaceConfigInStore = (next: ExternalProviderConfig): void => {
  const { configs } = store.getSnapshot();

  if (configs) {
    store.patchSnapshot({
      configs: configs.map((config) => (config.provider_id === next.provider_id ? next : config)),
    });
  }
};

/** Persist changes, then patch the shared snapshot; returns the new config. */
export const saveExternalProviderConfig = async (
  providerId: string,
  changes: { api_key?: string; base_url?: string | null }
): Promise<ExternalProviderConfig> => {
  const owner = captureAccountScope();
  const next = await setExternalProviderConfig(providerId, changes, owner.signal);

  if (isAccountScopeCurrent(owner)) {
    replaceConfigInStore(next);
  }

  return next;
};

/** Clear the provider's key and base URL, then patch the shared snapshot. */
export const clearExternalProviderConfig = async (providerId: string): Promise<ExternalProviderConfig> => {
  const owner = captureAccountScope();
  const next = await resetExternalProviderConfig(providerId, owner.signal);

  if (isAccountScopeCurrent(owner)) {
    replaceConfigInStore(next);
  }

  return next;
};

export const getExternalProvidersSnapshot = (): ExternalProvidersSnapshot => store.getSnapshot();

export const useExternalProvidersSelector = store.useSelector;

export const setExternalProvidersSnapshotForTests = (next: Partial<ExternalProvidersSnapshot>): void => {
  store.patchSnapshot(next);
};
