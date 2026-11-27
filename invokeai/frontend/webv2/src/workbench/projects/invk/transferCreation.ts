import type { ProjectCreateRequest } from '@workbench/projects/api';
import type { ProjectDocumentV3 } from '@workbench/projects/projectDocument';

import { applyAuthoritativeProjectBoard } from '@workbench/projects/projectDocument';
import { toProjectCreation } from '@workbench/projects/projectHalves';

/**
 * What an import or a duplication creates, in one request: the content beside the creator's workspace, so the project
 * never exists without it, or a whole document with the workspace inside (`toProjectCreation`). The workspace opens on
 * the inbox the create claims, which is why the inbox is staged before the create: its id must be known to go in.
 */
export const toTransferredProjectCreation = (
  document: ProjectDocumentV3,
  inboxStagingBoardId: string
): Pick<ProjectCreateRequest, 'board_id' | 'data' | 'minimum_document_schema_version' | 'workspace'> => {
  const creation = toProjectCreation(document);
  const openingOnInbox = (half: Record<string, unknown>): Record<string, unknown> =>
    applyAuthoritativeProjectBoard(half, inboxStagingBoardId, { selectBoard: true });

  return creation.workspace
    ? {
        board_id: inboxStagingBoardId,
        data: creation.data as unknown as Record<string, unknown>,
        minimum_document_schema_version: creation.documentSchemaVersion,
        workspace: openingOnInbox(creation.workspace as unknown as Record<string, unknown>),
      }
    : {
        board_id: inboxStagingBoardId,
        data: openingOnInbox(creation.data as unknown as Record<string, unknown>),
        minimum_document_schema_version: creation.documentSchemaVersion,
      };
};
