import { acquireExclusiveLock, type BrowserLockResult, hasWebLocks } from '@platform/browser/webLocks';

const LOCK_PREFIX = 'invokeai:v7:webv2:project-lifecycle';

export const getProjectLifecycleLockName = (storageSuffix: string, projectId: string): string =>
  `${LOCK_PREFIX}${storageSuffix}:${JSON.stringify(projectId)}`;

/**
 * The exclusive project-lifecycle lock: held, it means no tab has an active queue run for the project (each run holds
 * the name shared). `unavailable` without Web Locks.
 */
export const acquireProjectMutationLock = (
  storageSuffix: string,
  projectId: string,
  acquire: (name: string) => Promise<BrowserLockResult> = acquireExclusiveLock
): Promise<BrowserLockResult> => acquire(getProjectLifecycleLockName(storageSuffix, projectId));

/**
 * The lock policy for project delete, conflict discard and save as new. The backend rejects every write that would
 * conflict with them (revision checks, and 404 once a project is deleted), so without the Web Locks API (a plain-HTTP
 * origin) they run in single-tab mode: nothing is held, and the queue-run journal check is the only cross-tab guard
 * left. Where the API exists, other tabs can hold the lock, so a failed request stays `unavailable` and the caller
 * refuses. Queue-recovery discard has no backend fence and refuses `unavailable` either way.
 */
export const withSingleTabFallback = (lock: BrowserLockResult): BrowserLockResult =>
  lock.kind === 'unavailable' && !hasWebLocks() ? { kind: 'acquired', release: () => Promise.resolve() } : lock;
