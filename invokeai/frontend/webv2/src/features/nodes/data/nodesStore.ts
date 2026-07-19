import type { NodePackInfo } from '@features/nodes/core/catalog';

import { createExternalStore } from '@platform/state/externalStore';
import { getApiErrorMessage } from '@platform/transport/http';

import { listCustomNodePacks } from './api';

export interface CustomNodesSnapshot {
  nodePacks: NodePackInfo[];
  customNodesPath: string | null;
  status: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
}

const store = createExternalStore<CustomNodesSnapshot>({
  customNodesPath: null,
  error: null,
  nodePacks: [],
  status: 'idle',
});

let inflightRefresh: Promise<void> | null = null;

export const refreshCustomNodePacks = (): Promise<void> => {
  if (inflightRefresh) {
    return inflightRefresh;
  }

  store.patchSnapshot({ status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'loading' });

  inflightRefresh = listCustomNodePacks()
    .then((response) => {
      store.patchSnapshot({
        customNodesPath: response.customNodesPath,
        error: null,
        nodePacks: response.nodePacks,
        status: 'loaded',
      });
    })
    .catch((error: unknown) => {
      store.patchSnapshot({
        error: getApiErrorMessage(error, 'Failed to load custom node packs.'),
        status: store.getSnapshot().nodePacks.length > 0 ? 'loaded' : 'error',
      });
    })
    .finally(() => {
      inflightRefresh = null;
    });

  return inflightRefresh;
};

export const ensureCustomNodePacksLoaded = (): void => {
  if (store.getSnapshot().status === 'idle') {
    void refreshCustomNodePacks();
  }
};

export const removeCustomNodePackFromStore = (packName: string): void => {
  store.patchSnapshot({ nodePacks: store.getSnapshot().nodePacks.filter((pack) => pack.name !== packName) });
};

export const useCustomNodesSelector = store.useSelector;

export const useCustomNodesSnapshot = (): CustomNodesSnapshot => store.useSnapshot();
