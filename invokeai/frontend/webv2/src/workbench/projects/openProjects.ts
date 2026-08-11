import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createSingleFlight } from '@platform/state/singleFlight';

import { fetchSessionBlob } from './session';

/**
 * Which projects the editor currently has open, read from the saved session.
 *
 * The Launchpad asks this from two places — the top bar's way back in, and the
 * library's "open" grouping — so it lives in one store with one request rather
 * than a fetch per component.
 *
 * `ids: null` means *unknown*, not empty: a first run, a session blob written
 * before the open set existed, or an unreachable backend all land there. The
 * `/app` guard makes the same distinction and declines to redirect on it, so
 * callers must not treat unknown as "nothing is open".
 */

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
