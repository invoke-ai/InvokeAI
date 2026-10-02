import type { AccountScope } from '@platform/state/accountLifecycle';

import { captureAccountScope } from '@platform/state/accountLifecycle';

import {
  createDurableSyncedWorkbenchPersistence,
  type DurableHydratedWorkbenchSnapshot,
  type DurableSyncedWorkbenchPersistence,
  type DurableWorkbenchSaveResult,
  type ProjectBoardAssignment,
  type ProjectConflictInfo,
  type RecoverableProjectDraft,
} from './durableSyncedPersistence';

export { serializeProjectDocument } from './projectDocument';
export { deserializeProjectDocument, deserializeProjectRecord } from './projectHydration';
export { ProjectDocumentTooLargeError, WorkbenchBackendUnavailableError } from './durableSyncedPersistence';
export type { DurableHydratedWorkbenchSnapshot, ProjectBoardAssignment, ProjectConflictInfo, RecoverableProjectDraft };

export type WorkbenchSaveResult = DurableWorkbenchSaveResult;
export type SyncedWorkbenchPersistence = DurableSyncedWorkbenchPersistence;

export interface WorkbenchLoadOptions {
  createNew?: boolean;
  openProjectId?: string;
}

export const createSyncedWorkbenchPersistence = (
  owner: AccountScope = captureAccountScope()
): SyncedWorkbenchPersistence => createDurableSyncedWorkbenchPersistence(owner);

export const clearAllWorkbenchData = async (owner: AccountScope = captureAccountScope()): Promise<void> => {
  await createDurableSyncedWorkbenchPersistence(owner).clearWorkbench();
};
