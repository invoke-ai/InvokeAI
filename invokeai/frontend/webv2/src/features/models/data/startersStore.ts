import type { StarterModelResponse } from '@features/models/core/types';

import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import { getStarterModels } from './api';

/**
 * Cached starter-models catalog. The list is large and rarely changes, so it
 * is fetched once and served from memory on revisits; installs revalidate it
 * in the background (the `is_installed` flags shift) without blanking the UI.
 */

export interface StartersSnapshot {
  response: StarterModelResponse | null;
  status: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
}

const EMPTY_STARTERS_SNAPSHOT: StartersSnapshot = { error: null, response: null, status: 'idle' };
const store = createExternalStore<StartersSnapshot>(EMPTY_STARTERS_SNAPSHOT);

let inflightRefresh: Promise<void> | null = null;

registerAccountOwnedResource({
  clear: () => {
    inflightRefresh = null;
    store.setSnapshot(EMPTY_STARTERS_SNAPSHOT);
  },
  name: 'starter-models',
});

export const refreshStarters = (): Promise<void> => {
  if (inflightRefresh) {
    return inflightRefresh;
  }

  const owner = captureAccountScope();
  store.patchSnapshot({ status: store.getSnapshot().response ? 'loaded' : 'loading' });

  const refresh = getStarterModels(owner.signal)
    .then((response) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      store.patchSnapshot({ error: null, response, status: 'loaded' });
    })
    .catch((error: unknown) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      store.patchSnapshot({
        error: error instanceof Error ? error.message : 'Failed to load starter models.',
        status: store.getSnapshot().response ? 'loaded' : 'error',
      });
    })
    .finally(() => {
      if (inflightRefresh === refresh) {
        inflightRefresh = null;
      }
    });

  inflightRefresh = refresh;
  return inflightRefresh;
};

export const ensureStartersLoaded = (): void => {
  if (store.getSnapshot().status === 'idle') {
    void refreshStarters();
  }
};

/** Revalidate installed flags after an install lands, if already loaded. */
export const refreshStartersIfLoaded = (): void => {
  if (store.getSnapshot().status === 'loaded') {
    void refreshStarters();
  }
};

export const useStartersSelector = store.useSelector;

export const useStartersSnapshot = (): StartersSnapshot => store.useSnapshot();
