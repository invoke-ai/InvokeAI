import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import type { BoardDTO } from 'services/api/types';

/**
 * Returns permission flags for the given board based on the current user:
 * - `canWriteImages`: can add / delete images in the board
 *   (owner or admin always; non-owner allowed only for public boards)
 * - `canRenameBoard`: can rename the board (owner or admin only)
 * - `canDeleteBoard`: can delete the board (owner or admin only)
 *
 * When `board` is null/undefined (e.g. "uncategorized"), all permissions are
 * granted so that existing behaviour is preserved.
 *
 * When `currentUser` is null the app is running without authentication
 * (single-user mode), so full access is granted unconditionally.
 */
export const useBoardAccess = (board: BoardDTO | null | undefined) => {
  const currentUser = useAppSelector(selectCurrentUser);

  if (!board) {
    return { canWriteImages: true, canRenameBoard: true, canDeleteBoard: true };
  }

  const isOwnerOrAdmin = !currentUser || currentUser.is_admin || board.user_id === currentUser.user_id;

  return {
    canWriteImages: isOwnerOrAdmin || board.board_visibility === 'public',
    canRenameBoard: isOwnerOrAdmin,
    canDeleteBoard: isOwnerOrAdmin,
  };
};
