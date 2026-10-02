import { createLogger } from '@platform/logging/logger';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { apiFetchJson } from '@platform/transport/http';

const logger = createLogger({ area: 'whats-new', namespace: 'app' });

export interface WhatsNewSnapshot {
  /**
   * Closed during this page load. A version that arrives after a manual close must not reopen the notes the user
   * just dismissed; the next load still shows an unseen version.
   */
  isDismissed: boolean;
  /** Opened from the app menu; the automatic first-run showing is derived from preferences instead. */
  isRequested: boolean;
  /**
   * The server version, or null while unknown. Held here rather than in the query cache: the gate lives above
   * the router, and account transitions clear that cache without remounting it.
   */
  version: string | null;
}

const store = createExternalStore<WhatsNewSnapshot>({ isDismissed: false, isRequested: false, version: null });

let versionLoad: Promise<void> | null = null;

// The server version is not account data; only the request to show the notes follows the account.
registerAccountOwnedResource({
  clear: () => store.patchSnapshot({ isDismissed: false, isRequested: false }),
  name: 'whats-new',
});

/** The version only changes across a server restart, so one successful read per page load is enough. */
export const loadAppVersion = (): Promise<void> => {
  if (store.getSnapshot().version !== null) {
    return Promise.resolve();
  }

  versionLoad ??= apiFetchJson<{ version: string }>('/api/v1/app/version')
    .then((response) => store.patchSnapshot({ version: response.version }))
    .catch((error: unknown) => {
      // The notes stay closed until a later load succeeds; nothing else depends on them.
      logger.warn({ error, message: 'Failed to load the app version', name: 'app.whats-new.version-failed' });
    })
    .finally(() => {
      versionLoad = null;
    });

  return versionLoad;
};

export const openWhatsNew = (): void => {
  store.patchSnapshot({ isRequested: true });
  // A failed boot read leaves the version out of the header and links; try again while the notes are open.
  void loadAppVersion();
};

export const dismissWhatsNew = (): void => store.patchSnapshot({ isDismissed: true, isRequested: false });

export const getWhatsNewSnapshot = (): WhatsNewSnapshot => store.getSnapshot();

export const useWhatsNewSnapshot = (): WhatsNewSnapshot => store.useSnapshot();
