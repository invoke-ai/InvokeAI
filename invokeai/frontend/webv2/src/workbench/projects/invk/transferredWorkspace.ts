import { type AccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import {
  type ProjectRecordDTO,
  type ProjectWithWorkspaceDTO,
  saveFirstProjectWorkspace,
} from '@workbench/projects/api';
import { applyAuthoritativeProjectBoard } from '@workbench/projects/projectDocument';

import { isRequestCancellation } from './assetTransport';

/**
 * Saves the importing or duplicating account's workspace beside the content just created, opening on the inbox the
 * server says the project claimed, which only the created record names. Never fatal, but reported: without it the
 * project opens on the workspace a new one starts with.
 */
export const saveTransferredWorkspace = async (
  record: ProjectRecordDTO,
  workspace: Record<string, unknown>,
  owner: AccountScope
): Promise<{ record: ProjectWithWorkspaceDTO; workspaceNotSaved: boolean }> => {
  try {
    const saved = await saveFirstProjectWorkspace(
      record,
      applyAuthoritativeProjectBoard(workspace, record.board_id, { selectBoard: true }),
      owner.signal
    );
    return { record: saved, workspaceNotSaved: false };
  } catch (error) {
    if (isRequestCancellation(error) || !isAccountScopeCurrent(owner)) {
      throw error;
    }
    return { record: { ...record, workspace: null }, workspaceNotSaved: true };
  }
};
