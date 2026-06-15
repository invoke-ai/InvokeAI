import { createExternalStore } from '@workbench/externalStore';

import type { StarterModelResponse } from './types';

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

const store = createExternalStore<StartersSnapshot>({ error: null, response: null, status: 'idle' });

let inflightRefresh: Promise<void> | null = null;

export const refreshStarters = (): Promise<void> => {
  if (inflightRefresh) {
    return inflightRefresh;
  }

  store.patchSnapshot({ status: store.getSnapshot().response ? 'loaded' : 'loading' });

  inflightRefresh = getStarterModels()
    .then((response) => {
      store.patchSnapshot({ error: null, response, status: 'loaded' });
    })
    .catch((error: unknown) => {
      store.patchSnapshot({
        error: error instanceof Error ? error.message : 'Failed to load starter models.',
        status: store.getSnapshot().response ? 'loaded' : 'error',
      });
    })
    .finally(() => {
      inflightRefresh = null;
    });

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

export const useStartersSnapshot = (): StartersSnapshot => store.useSnapshot();
