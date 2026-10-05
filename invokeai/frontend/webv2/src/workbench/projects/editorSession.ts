import { createUuid } from '@platform/browser/randomUuid';
import { acquireExclusiveLock, type ExclusiveLockResult } from '@platform/browser/webLocks';

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

export const createEditorSessionProvider = (
  storage: SessionStoragePort,
  acquireLock: AcquireLock = acquireExclusiveLock,
  createId: () => string = createUuid
): (() => Promise<EditorSession>) => {
  let sessionPromise: Promise<EditorSession> | null = null;
  let currentSession: EditorSession | null = null;

  const persist = (id: string): void => {
    try {
      storage.setItem(EDITOR_SESSION_STORAGE_KEY, id);
    } catch {
      return;
    }
  };

  const claim = async (): Promise<EditorSession> => {
    let persistedId: string | null = null;
    try {
      persistedId = storage.getItem(EDITOR_SESSION_STORAGE_KEY);
    } catch {
      persistedId = null;
    }

    let candidate = persistedId && persistedId.length <= 128 ? persistedId : createId();
    for (;;) {
      const result = await acquireLock(`${EDITOR_SESSION_LOCK_PREFIX}${candidate}`);
      if (result.kind === 'acquired') {
        persist(candidate);
        const session: EditorSession = {
          id: candidate,
          async release() {
            if (currentSession === session) {
              currentSession = null;
              sessionPromise = null;
            }
            await result.release();
          },
        };
        currentSession = session;
        return session;
      }
      candidate = createId();
      if (result.kind === 'unavailable') {
        persist(candidate);
        const session: EditorSession = {
          id: candidate,
          release() {
            if (currentSession === session) {
              currentSession = null;
              sessionPromise = null;
            }
            return Promise.resolve();
          },
        };
        currentSession = session;
        return session;
      }
    }
  };

  return () => {
    sessionPromise ??= claim();
    return sessionPromise;
  };
};

export const getEditorSession = createEditorSessionProvider({
  getItem: (key) => window.sessionStorage.getItem(key),
  setItem: (key, value) => window.sessionStorage.setItem(key, value),
});
