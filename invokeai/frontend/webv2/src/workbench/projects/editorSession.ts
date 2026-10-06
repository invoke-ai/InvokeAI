import { createUuid } from '@platform/browser/randomUuid';
import { acquireExclusiveLock, isLockHeld, type ExclusiveLockResult } from '@platform/browser/webLocks';

export const EDITOR_SESSION_STORAGE_KEY = 'invokeai:v7:webv2:editor-session';
const EDITOR_SESSION_LOCK_PREFIX = 'invokeai:v7:webv2:editor-session:';

interface SessionStoragePort {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export interface EditorSession {
  id: string;
  release(): Promise<void>;
}

type AcquireLock = (name: string) => Promise<ExclusiveLockResult>;

/**
 * A reload may claim its session before the old document has let the lock go, and a duplicated tab (copied session
 * storage) never gets it. The persisted id is retried this long before a session of its own is taken, so a reload
 * keeps its session, and with it its own unload journal entries, at the cost of a short wait for a duplicated tab.
 */
const PERSISTED_CLAIM_RETRIES = 5;
const PERSISTED_CLAIM_RETRY_MS = 100;

const waitFor = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

/**
 * Whether a page still holds the editor session: its lock is released only when its last holder releases it or the
 * page goes away. False when it cannot be told (no Web Locks), so recovery then proceeds as if the page were gone.
 */
export const isEditorSessionLive = async (
  editorSessionId: string,
  queryLock: (name: string) => Promise<boolean | null> = isLockHeld
): Promise<boolean> => (await queryLock(`${EDITOR_SESSION_LOCK_PREFIX}${editorSessionId}`)) === true;

/**
 * Each call is one holder of the tab's editor session; the lock is given back when the last holder releases. A
 * superseded editor that is still finishing its exit and the editor that replaced it share one claim, so neither can
 * drop the lock from under the other.
 */
export const createEditorSessionProvider = (
  storage: SessionStoragePort,
  acquireLock: AcquireLock = acquireExclusiveLock,
  createId: () => string = createUuid,
  wait: (ms: number) => Promise<void> = waitFor
): (() => Promise<EditorSession>) => {
  let shared: { claim: Promise<{ id: string; release(): Promise<void> }>; holders: number } | null = null;

  const persist = (id: string): void => {
    try {
      storage.setItem(EDITOR_SESSION_STORAGE_KEY, id);
    } catch {
      return;
    }
  };

  const claim = async (): Promise<{ id: string; release(): Promise<void> }> => {
    let persistedId: string | null = null;
    try {
      persistedId = storage.getItem(EDITOR_SESSION_STORAGE_KEY);
    } catch {
      persistedId = null;
    }

    let candidate = persistedId && persistedId.length <= 128 ? persistedId : createId();
    let persistedRetries = candidate === persistedId ? PERSISTED_CLAIM_RETRIES : 0;
    for (;;) {
      const result = await acquireLock(`${EDITOR_SESSION_LOCK_PREFIX}${candidate}`);
      if (result.kind === 'acquired') {
        persist(candidate);
        return { id: candidate, release: result.release };
      }
      if (result.kind === 'contended' && persistedRetries > 0) {
        persistedRetries -= 1;
        await wait(PERSISTED_CLAIM_RETRY_MS);
        continue;
      }
      candidate = createId();
      if (result.kind === 'unavailable') {
        persist(candidate);
        return { id: candidate, release: () => Promise.resolve() };
      }
    }
  };

  return () => {
    shared ??= { claim: claim(), holders: 0 };
    const current = shared;
    current.holders += 1;
    return current.claim.then(({ id, release }) => {
      let isReleased = false;
      return {
        id,
        async release() {
          if (isReleased) {
            return;
          }
          isReleased = true;
          current.holders -= 1;
          if (current.holders > 0) {
            return;
          }
          // Stop publishing the claim before the lock is given back, so no new holder joins a released one.
          if (shared === current) {
            shared = null;
          }
          await release();
        },
      };
    });
  };
};

export const getEditorSession = createEditorSessionProvider({
  getItem: (key) => window.sessionStorage.getItem(key),
  setItem: (key, value) => window.sessionStorage.setItem(key, value),
});
