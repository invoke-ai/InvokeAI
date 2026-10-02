import type { AccountScope } from '@platform/state/accountLifecycle';

import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { acquireProjectMutationLock } from '@workbench/projects/projectLifecycleLocks';

import { createAccountOwnedQueueRunJournal, type QueueRunJournal } from './queueRunJournal';

export const QUEUE_RECOVERY_EXPORT_FORMAT = 'invokeai-webv2-queue-run-recovery';
export const QUEUE_RECOVERY_EXPORT_SCHEMA_VERSION = 1;

export type QueueRecoveryProjectIdsResult = { kind: 'available'; projectIds: string[] } | { kind: 'unavailable' };

export type QueueRecoveryErrorCode = 'lock-contended' | 'lock-unavailable' | 'storage-unavailable';

export class QueueRecoveryError extends Error {
  constructor(readonly code: QueueRecoveryErrorCode) {
    super(code);
    this.name = 'QueueRecoveryError';
  }
}

interface QueueRecoveryDependencies {
  acquireLock?: typeof acquireProjectMutationLock;
  journal?: (owner: AccountScope) => Promise<QueueRunJournal>;
  now?: () => number;
}

const openJournal = (owner: AccountScope, dependencies: QueueRecoveryDependencies): Promise<QueueRunJournal> =>
  (dependencies.journal ?? createAccountOwnedQueueRunJournal)(owner);

export const listQueueRecoveryProjectIds = async (
  owner: AccountScope,
  dependencies: QueueRecoveryDependencies = {}
): Promise<QueueRecoveryProjectIdsResult> => {
  let journal: QueueRunJournal | null = null;
  try {
    assertAccountScopeCurrent(owner);
    journal = await openJournal(owner, dependencies);
    assertAccountScopeCurrent(owner);
    const result = await journal.listProjectIds();
    assertAccountScopeCurrent(owner);
    return result.kind === 'available'
      ? { kind: 'available', projectIds: [...new Set(result.projectIds)].sort() }
      : result;
  } catch {
    return { kind: 'unavailable' };
  } finally {
    journal?.close();
  }
};

export const createQueueRecoveryExport = async (
  owner: AccountScope,
  projectId: string,
  dependencies: QueueRecoveryDependencies = {}
): Promise<string> => {
  let journal: QueueRunJournal | null = null;
  try {
    assertAccountScopeCurrent(owner);
    journal = await openJournal(owner, dependencies);
    assertAccountScopeCurrent(owner);
    const result = await journal.listForProject(projectId);
    assertAccountScopeCurrent(owner);
    if (result.kind === 'unavailable') {
      throw new QueueRecoveryError('storage-unavailable');
    }
    return JSON.stringify({
      exportedAt: new Date((dependencies.now ?? Date.now)()).toISOString(),
      format: QUEUE_RECOVERY_EXPORT_FORMAT,
      projectId,
      runs: result.entries.map(({ item, queueItemId, submissionOrder }) => ({ item, queueItemId, submissionOrder })),
      schemaVersion: QUEUE_RECOVERY_EXPORT_SCHEMA_VERSION,
    });
  } finally {
    journal?.close();
  }
};

export const discardQueueRecoveryProject = async (
  owner: AccountScope,
  projectId: string,
  dependencies: QueueRecoveryDependencies = {}
): Promise<void> => {
  assertAccountScopeCurrent(owner);
  const lock = await (dependencies.acquireLock ?? acquireProjectMutationLock)(owner.storageSuffix, projectId);
  if (lock.kind === 'contended') {
    throw new QueueRecoveryError('lock-contended');
  }
  if (lock.kind === 'unavailable') {
    throw new QueueRecoveryError('lock-unavailable');
  }

  let journal: QueueRunJournal | null = null;
  try {
    assertAccountScopeCurrent(owner);
    journal = await openJournal(owner, dependencies);
    assertAccountScopeCurrent(owner);
    const result = await journal.deleteForProject(projectId);
    assertAccountScopeCurrent(owner);
    if (result.kind === 'unavailable') {
      throw new QueueRecoveryError('storage-unavailable');
    }
  } finally {
    journal?.close();
    await lock.release();
  }
};

export const getQueueRecoveryExportName = (projectId: string): string => {
  const safeProjectId = projectId
    .trim()
    .replaceAll(/[^\w.-]+/g, '-')
    .replaceAll(/^-+|-+$/g, '')
    .slice(0, 96);
  return `${safeProjectId || 'project'}.queue-recovery.json`;
};
