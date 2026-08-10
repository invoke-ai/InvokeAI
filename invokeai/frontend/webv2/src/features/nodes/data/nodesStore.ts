import type { NodePackInfo } from '@features/nodes/core/catalog';

import {
  type AccountScope,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createTrailingSingleFlight } from '@platform/state/singleFlight';
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

const refreshFlight = createTrailingSingleFlight();

registerAccountOwnedResource({
  clear: () => {
    refreshFlight.reset();
    store.setSnapshot(EMPTY_CUSTOM_NODES_SNAPSHOT);
  },
  name: 'custom-node-packs',
});

/** Re-fetch the pack list; concurrent calls share one request, and a call made mid-flight queues one trailing rerun. */
export const refreshCustomNodePacks = (owner: AccountScope = captureAccountScope()): Promise<void> =>
  refreshFlight.run(() => {
    store.patchSnapshot({ status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'loading' });

    return listCustomNodePacks(owner.signal)
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
      });
  });

/** Fetch on first use or retry after an error; callers share and can await the request. */
export const ensureCustomNodePacksLoaded = (): Promise<void> => {
  const { status } = store.getSnapshot();

  if (status === 'idle' || status === 'error') {
    return refreshCustomNodePacks();
  }

  return refreshFlight.inflight() ?? Promise.resolve();
};

export const getCustomNodesSnapshot = (): CustomNodesSnapshot => store.getSnapshot();

export const removeCustomNodePackFromStore = (packName: string): void => {
  store.patchSnapshot({ nodePacks: store.getSnapshot().nodePacks.filter((pack) => pack.name !== packName) });
};

export const useCustomNodesSelector = store.useSelector;

export const useCustomNodesSnapshot = (): CustomNodesSnapshot => store.useSnapshot();
