export type BrowserLockResult =
  | { kind: 'acquired'; release(): Promise<void> }
  | { kind: 'contended' }
  | { kind: 'unavailable' };

const acquireLock = (
  name: string,
  mode: 'exclusive' | 'shared',
  lockManager?: LockManager
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
        .request(name, { ifAvailable: true, mode }, async (lock) => {
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
