import type { GalleryBoard, GalleryBoardAction } from './types';

/*
 * The caller's relation to a board, from what the server reports. The server alone decides who may do what; these
 * only read its answer. A board without `access` (a virtual board) is the caller's own and allows what it always did.
 */

/** Whether the server lets the caller take an action on the board. */
export const canOnBoard = (board: GalleryBoard, action: GalleryBoardAction): boolean =>
  board.access === undefined || board.access.actions.includes(action);

/** The caller's own board (or a virtual one). */
export const isOwnGalleryBoard = (board: GalleryBoard): boolean =>
  board.access === undefined || board.access.role === 'owner';

/** Someone else's board the caller has a role on. A real role wins over admin rights. */
export const isSharedGalleryBoard = (board: GalleryBoard): boolean =>
  board.access !== undefined && board.access.role !== null && board.access.role !== 'owner';

/**
 * The boards as a server with one account sees them: every board is the caller's own. Boards other accounts made
 * while the server ran with several come back as an admin's view of someone else's; there is no one else to share
 * with or to name, so they read as the caller's, as they did before sharing.
 */
export const asSingleAccountBoards = (boards: readonly GalleryBoard[]): GalleryBoard[] =>
  boards.map((board) =>
    board.access === undefined || board.access.role === 'owner'
      ? board
      : { ...board, access: { ...board.access, adminOverride: false, role: 'owner' } }
  );

/** Someone else's board the caller sees only because they are an admin. */
export const isOtherUsersGalleryBoard = (board: GalleryBoard): boolean =>
  board.access !== undefined && board.access.role === null;
