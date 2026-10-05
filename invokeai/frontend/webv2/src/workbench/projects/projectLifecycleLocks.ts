import { acquireExclusiveLock, type BrowserLockResult } from '@platform/browser/webLocks';

const LOCK_PREFIX = 'invokeai:v7:webv2:project-lifecycle';

export const getProjectLifecycleLockName = (storageSuffix: string, projectId: string): string =>
  `${LOCK_PREFIX}${storageSuffix}:${JSON.stringify(projectId)}`;

export const acquireProjectMutationLock = (
  storageSuffix: string,
  projectId: string,
  acquire: (name: string) => Promise<BrowserLockResult> = acquireExclusiveLock
): Promise<BrowserLockResult> => acquire(getProjectLifecycleLockName(storageSuffix, projectId));
