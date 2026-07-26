import type { NodePackInfo } from '@features/nodes/core/catalog';

import {
  type AccountScope,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { getApiErrorMessage } from '@platform/transport/http';

import { listCustomNodePacks } from './api';

export interface CustomNodesSnapshot {
  nodePacks: NodePackInfo[];
  customNodesPath: string | null;
  status: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
}

const EMPTY_CUSTOM_NODES_SNAPSHOT: CustomNodesSnapshot = {
  customNodesPath: null,
  error: null,
  nodePacks: [],
  status: 'idle',
};
const store = createExternalStore<CustomNodesSnapshot>(EMPTY_CUSTOM_NODES_SNAPSHOT);

let inflightRefresh: Promise<void> | null = null;

registerAccountOwnedResource({
  clear: () => {
    inflightRefresh = null;
    store.setSnapshot(EMPTY_CUSTOM_NODES_SNAPSHOT);
  },
  name: 'custom-node-packs',
});

export const refreshCustomNodePacks = (owner: AccountScope = captureAccountScope()): Promise<void> => {
  if (inflightRefresh) {
    return inflightRefresh;
  }

  store.patchSnapshot({ status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'loading' });

  const refresh = listCustomNodePacks(owner.signal)
    .then((response) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      store.patchSnapshot({
        customNodesPath: response.customNodesPath,
        error: null,
        nodePacks: response.nodePacks,
        status: 'loaded',
      });
    })
    .catch((error: unknown) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      store.patchSnapshot({
        error: getApiErrorMessage(error, 'Failed to load custom node packs.'),
        status: store.getSnapshot().nodePacks.length > 0 ? 'loaded' : 'error',
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
