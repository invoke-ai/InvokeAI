import type { ContextMenuProps } from '@invoke-ai/ui-library';
import { ContextMenu, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { autoAddBoardIdChanged, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArchiveBold, PiArchiveFill, PiDownloadBold, PiPlusBold } from 'react-icons/pi';
import { useUpdateBoardMutation } from 'services/api/endpoints/boards';
import { useBulkDownloadImagesMutation } from 'services/api/endpoints/images';
import { useBoardName } from 'services/api/hooks/useBoardName';
import type { BoardDTO } from 'services/api/types';

import GalleryBoardContextMenuItems from './GalleryBoardContextMenuItems';

type Props = {
  board: BoardDTO;
  children: ContextMenuProps<HTMLDivElement>['children'];
  setBoardToDelete: (board?: BoardDTO) => void;
};

const BoardContextMenu = ({ board, setBoardToDelete, children }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoAssignBoardOnClick = useAppSelector((s) => s.gallery.autoAssignBoardOnClick);
  const selectIsSelectedForAutoAdd = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => board.board_id === gallery.autoAddBoardId),
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

  const handleArchive = useCallback(() => {
    updateBoard({
      board_id: board.board_id,
      changes: { archived: true },
    });
  }, [board.board_id, updateBoard]);

  const handleUnarchive = useCallback(() => {
    updateBoard({
      board_id: board.board_id,
      changes: { archived: false },
    });
  }, [board.board_id, updateBoard]);

  const renderMenuFunc = useCallback(
    () => (
      <MenuList visibility="visible">
        <MenuGroup title={boardName}>
          <MenuItem
            icon={<PiPlusBold />}
            isDisabled={isSelectedForAutoAdd || autoAssignBoardOnClick || Boolean(board?.archived)}
            onClick={handleSetAutoAdd}
          >
            {t('boards.menuItemAutoAdd')}
          </MenuItem>
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
            <MenuItem icon={<PiArchiveFill />} onClick={handleArchive} isDisabled={isSelectedForAutoAdd}>
              {t('boards.archiveBoard')}
            </MenuItem>
          )}

          <GalleryBoardContextMenuItems board={board} setBoardToDelete={setBoardToDelete} />
        </MenuGroup>
      </MenuList>
    ),
    [
      autoAssignBoardOnClick,
      board,
      boardName,
      handleBulkDownload,
      handleSetAutoAdd,
      isBulkDownloadEnabled,
      isSelectedForAutoAdd,
      setBoardToDelete,
      t,
      handleArchive,
      handleUnarchive,
    ]
  );

  return <ContextMenu renderMenu={renderMenuFunc}>{children}</ContextMenu>;
};

export default memo(BoardContextMenu);
