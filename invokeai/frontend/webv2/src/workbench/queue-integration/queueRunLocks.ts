import type { QueueRunLockPort } from '@features/queue';

import {
  acquireExclusiveLock,
  acquireSharedLock,
  type BrowserLockResult,
  type ExclusiveLockResult,
  isLockHeld,
  waitForSharedLock,
} from '@platform/browser/webLocks';
import { getProjectLifecycleLockName } from '@workbench/projects/projectLifecycleLocks';

const LOCK_PREFIX = 'invokeai:v7:webv2:queue-run';

export const getQueueRunLockName = (storageSuffix: string, projectId: string, queueItemId: string): string =>
  `${LOCK_PREFIX}${storageSuffix}:${JSON.stringify([projectId, queueItemId])}`;

export const createQueueRunLockPort = (
  storageSuffix: string,
  acquireRun: (name: string) => Promise<ExclusiveLockResult> = acquireExclusiveLock,
  acquireProject: (name: string) => Promise<BrowserLockResult> = acquireSharedLock,
  isHeld: (name: string) => Promise<boolean | null> = isLockHeld,
  waitShared: (name: string, signal: AbortSignal) => Promise<BrowserLockResult> = waitForSharedLock
): QueueRunLockPort => ({
  isHeld: (projectId, queueItemId) => isHeld(getQueueRunLockName(storageSuffix, projectId, queueItemId)),
  async waitForRunRelease(projectId, queueItemId, signal) {
    // Shared, so tabs waiting on the same run never queue behind one another.
    const lock = await waitShared(getQueueRunLockName(storageSuffix, projectId, queueItemId), signal);
    if (lock.kind === 'acquired') {
      await lock.release();
    }
  },
  async acquire(projectId, queueItemId) {
    const projectLock = await acquireProject(getProjectLifecycleLockName(storageSuffix, projectId));
    if (projectLock.kind !== 'acquired') {
      return projectLock;
    }
    let runLock: ExclusiveLockResult;
    try {
      runLock = await acquireRun(getQueueRunLockName(storageSuffix, projectId, queueItemId));
    } catch (error) {
      await projectLock.release();
      throw error;
    }
    if (runLock.kind !== 'acquired') {
      await projectLock.release();
      return runLock;
    }
    return {
      kind: 'acquired',
      async release() {
        try {
          await runLock.release();
        } finally {
          await projectLock.release();
        }
      },
    };
  },
});
