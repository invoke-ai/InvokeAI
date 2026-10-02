import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createSingleFlight } from '@platform/state/singleFlight';

import { fetchSessionBlob } from './session';

/** Share the route guard's request and null-as-unknown contract. */

export interface OpenProjectsSnapshot {
  status: 'idle' | 'loading' | 'ready';
  ids: string[] | null;
  activeId: string | null;
}

const EMPTY_OPEN_PROJECTS: OpenProjectsSnapshot = { activeId: null, ids: null, status: 'idle' };
const store = createExternalStore<OpenProjectsSnapshot>(EMPTY_OPEN_PROJECTS);

registerAccountOwnedResource({
  clear: () => {
    store.setSnapshot(EMPTY_OPEN_PROJECTS);
  },
  name: 'open-projects',
});

export const useOpenProjectsSelector = store.useSelector;

export const getOpenProjects = (): OpenProjectsSnapshot => store.getSnapshot();

const refreshFlight = createSingleFlight<void>();

/** Re-read the session blob; concurrent calls share one request. */
export const refreshOpenProjects = (): Promise<void> => {
  const owner = captureAccountScope();

  if (store.getSnapshot().status === 'idle') {
    store.patchSnapshot({ status: 'loading' });
  }

  return refreshFlight.run(`open-projects:${owner.epoch}`, () =>
    fetchSessionBlob(owner.signal)
      .then((blob) => {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        store.setSnapshot({
          activeId: blob?.activeProjectId ?? null,
          ids: blob?.openProjectIds ?? null,
          status: 'ready',
        });
      })
      .catch(() => {
        if (isAccountScopeCurrent(owner)) {
          store.setSnapshot({ activeId: null, ids: null, status: 'ready' });
        }
      })
  );
};
