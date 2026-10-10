import type { InvkBoard } from './board';
import type { RestoredMediaLedger } from './restoreProjectMedia';
import type { InvkBoardIssue } from './transfer';

import { createStagingBoard, isRequestCancellation, placeBoardInProject } from './assetTransport';

/**
 * A project's boards travel as a set, but only the inbox rides the project create: the server claims it in the same
 * transaction that writes the document. So every board is staged alike — a private Library board, made in source
 * order so creation order survives — media lands on it before the commit point, and the others are moved into the
 * project once it exists.
 */

export interface StagedBoard {
  board: InvkBoard;
  /** The Library board standing in for `board` until the project exists. */
  stagingBoardId: string;
}

export interface MemberBoardPlacementDeps {
  placeBoardInProject?: typeof placeBoardInProject;
  signal?: AbortSignal;
}

/**
 * One staging board per board, in `boards` order, each on the ledger the moment it exists so a restore abandoned
 * part-way can delete it. The inbox's is named after the project, the others after themselves.
 */
export const createStagingBoards = async (
  boards: readonly InvkBoard[],
  projectName: string,
  ledger: RestoredMediaLedger,
  signal?: AbortSignal,
  create: typeof createStagingBoard = createStagingBoard
): Promise<StagedBoard[]> => {
  const staged: StagedBoard[] = [];

  for (const board of boards) {
    const stagingBoardId = await create(board.isInbox ? projectName : board.name, signal);

    ledger.boardIds.push(stagingBoardId);
    staged.push({ board, stagingBoardId });
  }

  return staged;
};

/** The staging board the project create claims as its inbox; `null` when nothing was staged. */
export const findInboxStagingBoardId = (staged: readonly StagedBoard[]): string | null =>
  staged.find(({ board }) => board.isInbox)?.stagingBoardId ?? null;

/**
 * Move every board but the inbox into the created project, archived as the source had it. Each settles on its own,
 * because a project with most of its boards beats none: a board that could not move is reported, and stays in the
 * Library with everything on it. Cancellation and account expiry still abort.
 */
export const placeMemberBoards = async (
  staged: readonly StagedBoard[],
  projectId: string,
  deps: MemberBoardPlacementDeps = {}
): Promise<InvkBoardIssue[]> => {
  const place = deps.placeBoardInProject ?? placeBoardInProject;
  const issues: InvkBoardIssue[] = [];

  for (const { board, stagingBoardId } of staged) {
    if (board.isInbox) {
      continue;
    }

    try {
      await place(stagingBoardId, projectId, board.archived, deps.signal);
    } catch (error) {
      if (isRequestCancellation(error)) {
        throw error;
      }

      issues.push({ name: board.name });
    }
  }

  return issues;
};
