import type { ContextMenuProps } from '@invoke-ai/ui-library';
import { ContextMenu, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { $boardToDelete } from 'features/gallery/components/Boards/DeleteBoardModal';
import { selectAutoAddBoardId, selectAutoAssignBoardOnClick } from 'features/gallery/store/gallerySelectors';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArchiveBold,
  PiArchiveFill,
  PiDownloadBold,
  PiGlobeBold,
  PiLockBold,
  PiPlusBold,
  PiShareNetworkBold,
  PiTrashSimpleBold,
} from 'react-icons/pi';
import { useUpdateBoardMutation } from 'services/api/endpoints/boards';
import { useBulkDownloadImagesMutation } from 'services/api/endpoints/images';
import { useBoardAccess } from 'services/api/hooks/useBoardAccess';
import { useBoardName } from 'services/api/hooks/useBoardName';
import type { BoardDTO } from 'services/api/types';

type Props = {
  board: BoardDTO;
  children: ContextMenuProps<HTMLDivElement>['children'];
};

const BoardContextMenu = ({ board, children }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoAssignBoardOnClick = useAppSelector(selectAutoAssignBoardOnClick);
  const currentUser = useAppSelector(selectCurrentUser);
  const selectIsSelectedForAutoAdd = useMemo(
    () => createSelector(selectAutoAddBoardId, (autoAddBoardId) => board.board_id === autoAddBoardId),
    [board.board_id]
  );

  const [updateBoard] = useUpdateBoardMutation();

  const isSelectedForAutoAdd = useAppSelector(selectIsSelectedForAutoAdd);
  const boardName = useBoardName(board.board_id);

  const [bulkDownload] = useBulkDownloadImagesMutation();

  // Only the board owner or admin can modify visibility
  const canChangeVisibility = currentUser !== null && (currentUser.is_admin || board.user_id === currentUser.user_id);

  const { canDeleteBoard } = useBoardAccess(board);

  const handleSetAutoAdd = useCallback(() => {
    dispatch(autoAddBoardIdChanged(board.board_id));
  }, [board.board_id, dispatch]);

  const handleBulkDownload = useCallback(() => {
    bulkDownload({ image_names: [], board_id: board.board_id });
  }, [board.board_id, bulkDownload]);

  const handleArchive = useCallback(async () => {
    try {
      await updateBoard({
        board_id: board.board_id,
        changes: { archived: true },
      }).unwrap();
    } catch {
      toast({
        status: 'error',
        title: 'Unable to archive board',
      });
    }
  }, [board.board_id, updateBoard]);

  const handleUnarchive = useCallback(() => {
    updateBoard({
      board_id: board.board_id,
      changes: { archived: false },
    });
  }, [board.board_id, updateBoard]);

  const handleSetVisibility = useCallback(
    async (visibility: 'private' | 'shared' | 'public') => {
      try {
        await updateBoard({
          board_id: board.board_id,
          changes: { board_visibility: visibility },
        }).unwrap();
      } catch {
        toast({ status: 'error', title: t('boards.updateBoardVisibilityError') });
      }
    },
    [board.board_id, t, updateBoard]
  );

  const handleSetVisibilityPrivate = useCallback(() => handleSetVisibility('private'), [handleSetVisibility]);

  const handleSetVisibilityShared = useCallback(() => handleSetVisibility('shared'), [handleSetVisibility]);

  const handleSetVisibilityPublic = useCallback(() => handleSetVisibility('public'), [handleSetVisibility]);

  const setAsBoardToDelete = useCallback(() => {
    $boardToDelete.set(board);
  }, [board]);

  const renderMenuFunc = useCallback(
    () => (
      <MenuList visibility="visible">
        <MenuGroup title={boardName}>
          {!autoAssignBoardOnClick && (
            <MenuItem icon={<PiPlusBold />} isDisabled={isSelectedForAutoAdd} onClick={handleSetAutoAdd}>
              {isSelectedForAutoAdd ? t('boards.selectedForAutoAdd') : t('boards.menuItemAutoAdd')}
            </MenuItem>
          )}

          <MenuItem icon={<PiDownloadBold />} onClickCapture={handleBulkDownload}>
            {t('boards.downloadBoard')}
          </MenuItem>

          {board.archived && (
            <MenuItem icon={<PiArchiveBold />} onClick={handleUnarchive} isDisabled={!canDeleteBoard}>
              {t('boards.unarchiveBoard')}
            </MenuItem>
          )}

          {!board.archived && (
            <MenuItem icon={<PiArchiveFill />} onClick={handleArchive} isDisabled={!canDeleteBoard}>
              {t('boards.archiveBoard')}
            </MenuItem>
          )}

          {canChangeVisibility && (
            <>
              <MenuItem
                icon={<PiLockBold />}
                onClick={handleSetVisibilityPrivate}
                isDisabled={board.board_visibility === 'private'}
              >
                {t('boards.setVisibilityPrivate')}
              </MenuItem>
              <MenuItem
                icon={<PiShareNetworkBold />}
                onClick={handleSetVisibilityShared}
                isDisabled={board.board_visibility === 'shared'}
              >
                {t('boards.setVisibilityShared')}
              </MenuItem>
              <MenuItem
                icon={<PiGlobeBold />}
                onClick={handleSetVisibilityPublic}
                isDisabled={board.board_visibility === 'public'}
              >
                {t('boards.setVisibilityPublic')}
              </MenuItem>
            </>
          )}

          <MenuItem
            color="error.300"
            icon={<PiTrashSimpleBold />}
            onClick={setAsBoardToDelete}
            isDestructive
            isDisabled={!canDeleteBoard}
          >
            {t('boards.deleteBoard')}
          </MenuItem>
        </MenuGroup>
      </MenuList>
    ),
    [
      boardName,
      autoAssignBoardOnClick,
      isSelectedForAutoAdd,
      handleSetAutoAdd,
      t,
      handleBulkDownload,
      board.archived,
      board.board_visibility,
      handleUnarchive,
      handleArchive,
      canChangeVisibility,
      handleSetVisibilityPrivate,
      handleSetVisibilityShared,
      handleSetVisibilityPublic,
      canDeleteBoard,
      setAsBoardToDelete,
    ]
  );

  return <ContextMenu renderMenu={renderMenuFunc}>{children}</ContextMenu>;
};

export default memo(BoardContextMenu);
