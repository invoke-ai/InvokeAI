export type BrowserLockResult =
  | { kind: 'acquired'; release(): Promise<void> }
  | { kind: 'contended' }
  | { kind: 'unavailable' };

const acquireLock = (
  name: string,
  mode: 'exclusive' | 'shared',
  lockManager?: LockManager,
  /** Wait for the lock until it is granted or this aborts, instead of taking it only when free. */
  waitSignal?: AbortSignal
): Promise<BrowserLockResult> => {
  let manager = lockManager;
  if (!manager) {
    try {
      manager = typeof navigator === 'undefined' ? undefined : navigator.locks;
    } catch {
      manager = undefined;
    }
  }
  if (!manager) {
    return Promise.resolve({ kind: 'unavailable' });
  }

  return new Promise((resolve) => {
    let didResolve = false;
    let releaseHold: () => void = () => undefined;
    const hold = new Promise<void>((release) => {
      releaseHold = release;
    });

    try {
      let request: Promise<void>;
      request = manager
        .request(name, waitSignal ? { mode, signal: waitSignal } : { ifAvailable: true, mode }, async (lock) => {
          if (!lock) {
            didResolve = true;
            resolve({ kind: 'contended' });
            return;
          }
          let isReleased = false;
          didResolve = true;
          resolve({
            kind: 'acquired',
            async release() {
              if (!isReleased) {
                isReleased = true;
                releaseHold();
              }
              await request;
            },
          });
          await hold;
        })
        .catch(() => {
          if (!didResolve) {
            didResolve = true;
            resolve({ kind: 'unavailable' });
          }
        });
    } catch {
      didResolve = true;
      resolve({ kind: 'unavailable' });
    }
  });
};

export type ExclusiveLockResult = BrowserLockResult;

export const acquireExclusiveLock = (name: string, lockManager?: LockManager): Promise<BrowserLockResult> =>
  acquireLock(name, 'exclusive', lockManager);

export const acquireSharedLock = (name: string, lockManager?: LockManager): Promise<BrowserLockResult> =>
  acquireLock(name, 'shared', lockManager);

/** Wait until the shared lock is granted; `unavailable` without the Web Locks API or once the signal aborts. */
export const waitForSharedLock = (
  name: string,
  signal: AbortSignal,
  lockManager?: LockManager
): Promise<BrowserLockResult> => acquireLock(name, 'shared', lockManager, signal);

/**
 * Whether any context of this origin currently holds the lock, read without requesting it, so a holder about to claim
 * the name is never turned away. `null` when the Web Locks API or its query is unavailable.
 */
export const isLockHeld = async (name: string, lockManager?: LockManager): Promise<boolean | null> => {
  let manager = lockManager;
  if (!manager) {
    try {
      manager = typeof navigator === 'undefined' ? undefined : navigator.locks;
    } catch {
      manager = undefined;
    }
  }
  if (!manager || typeof manager.query !== 'function') {
    return null;
  }
  try {
    const { held = [] } = await manager.query();
    return held.some((lock) => lock.name === name);
  } catch {
    return null;
  }
};
