import type { QueueRunLockPort } from '@features/queue';

import {
  acquireExclusiveLock,
  acquireSharedLock,
  type BrowserLockResult,
  type ExclusiveLockResult,
} from '@platform/browser/webLocks';
import { getProjectLifecycleLockName } from '@workbench/projects/projectLifecycleLocks';

const LOCK_PREFIX = 'invokeai:v7:webv2:queue-run';

export const getQueueRunLockName = (storageSuffix: string, projectId: string, queueItemId: string): string =>
  `${LOCK_PREFIX}${storageSuffix}:${JSON.stringify([projectId, queueItemId])}`;

export const createQueueRunLockPort = (
  storageSuffix: string,
  acquireRun: (name: string) => Promise<ExclusiveLockResult> = acquireExclusiveLock,
  acquireProject: (name: string) => Promise<BrowserLockResult> = acquireSharedLock
): QueueRunLockPort => ({
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
