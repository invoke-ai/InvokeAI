import type { ContextMenuProps } from '@invoke-ai/ui-library';
import { ContextMenu, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $boardToDelete } from 'features/gallery/components/Boards/DeleteBoardModal';
import { selectAutoAddBoardId, selectAutoAssignBoardOnClick } from 'features/gallery/store/gallerySelectors';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArchiveBold, PiArchiveFill, PiDownloadBold, PiPlusBold, PiTrashSimpleBold } from 'react-icons/pi';
import { useUpdateBoardMutation } from 'services/api/endpoints/boards';
import { useBulkDownloadImagesMutation } from 'services/api/endpoints/images';
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
  const selectIsSelectedForAutoAdd = useMemo(
    () => createSelector(selectAutoAddBoardId, (autoAddBoardId) => board.board_id === autoAddBoardId),
    [board.board_id]
  );

  const [updateBoard] = useUpdateBoardMutation();

  const isSelectedForAutoAdd = useAppSelector(selectIsSelectedForAutoAdd);
  const boardName = useBoardName(board.board_id);
  const isBulkDownloadEnabled = useFeatureStatus('bulkDownload');

  const [bulkDownload] = useBulkDownloadImagesMutation();

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
    } catch (error) {
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
          {isBulkDownloadEnabled && (
            <MenuItem icon={<PiDownloadBold />} onClickCapture={handleBulkDownload}>
              {t('boards.downloadBoard')}
            </MenuItem>
          )}

          {board.archived && (
            <MenuItem icon={<PiArchiveBold />} onClick={handleUnarchive}>
              {t('boards.unarchiveBoard')}
            </MenuItem>
          )}

          {!board.archived && (
            <MenuItem icon={<PiArchiveFill />} onClick={handleArchive}>
              {t('boards.archiveBoard')}
            </MenuItem>
          )}

          <MenuItem color="error.300" icon={<PiTrashSimpleBold />} onClick={setAsBoardToDelete} isDestructive>
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
      isBulkDownloadEnabled,
      handleBulkDownload,
      board.archived,
      handleUnarchive,
      handleArchive,
      setAsBoardToDelete,
    ]
  );

  return <ContextMenu renderMenu={renderMenuFunc}>{children}</ContextMenu>;
};

export default memo(BoardContextMenu);
