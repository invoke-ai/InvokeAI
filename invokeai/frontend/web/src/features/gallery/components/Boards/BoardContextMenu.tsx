import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { InvContextMenuProps } from 'common/components/InvContextMenu/InvContextMenu';
import { InvContextMenu } from 'common/components/InvContextMenu/InvContextMenu';
import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { InvMenuList } from 'common/components/InvMenu/InvMenuList';
import { InvMenuGroup } from 'common/components/InvMenu/wrapper';
import {
  autoAddBoardIdChanged,
  selectGallerySlice,
} from 'features/gallery/store/gallerySlice';
import type { BoardId } from 'features/gallery/store/types';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaDownload, FaPlus } from 'react-icons/fa';
import { useBulkDownloadImagesMutation } from 'services/api/endpoints/images';
import { useBoardName } from 'services/api/hooks/useBoardName';
import type { BoardDTO } from 'services/api/types';

import GalleryBoardContextMenuItems from './GalleryBoardContextMenuItems';
import { bulkDownloadRequested } from '../../store/actions';

type Props = {
  board?: BoardDTO;
  board_id: BoardId;
  children: InvContextMenuProps<HTMLDivElement>['children'];
  setBoardToDelete?: (board?: BoardDTO) => void;
};

const BoardContextMenu = ({
  board,
  board_id,
  setBoardToDelete,
  children,
}: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoAssignBoardOnClick = useAppSelector(
    (s) => s.gallery.autoAssignBoardOnClick
  );
  const selectIsSelectedForAutoAdd = useMemo(
    () =>
      createSelector(
        selectGallerySlice,
        (gallery) => board && board.board_id === gallery.autoAddBoardId
      ),
    [board]
  );

  const isSelectedForAutoAdd = useAppSelector(selectIsSelectedForAutoAdd);
  const boardName = useBoardName(board_id);
  const isBulkDownloadEnabled =
    useFeatureStatus('bulkDownload').isFeatureEnabled;

  const [bulkDownload] = useBulkDownloadImagesMutation();

  const handleSetAutoAdd = useCallback(() => {
    dispatch(autoAddBoardIdChanged(board_id));
  }, [board_id, dispatch]);

  const handleBulkDownload = useCallback(async () => {
    try {
      const response = await bulkDownload({
        image_names: [],
        board_id: board_id,
      }).unwrap();

      dispatch(bulkDownloadRequested({ type: 'board' }));

      dispatch(
        addToast({
          title: t('gallery.preparingDownload'),
          status: 'success',
          ...(response.response
            ? {
                description: response.response,
                duration: null,
                isClosable: true,
              }
            : {}),
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('gallery.preparingDownloadFailed'),
          status: 'error',
        })
      );
    }
  }, [t, board_id, bulkDownload, dispatch]);

  const renderMenuFunc = useCallback(
    () => (
      <InvMenuList visibility="visible">
        <InvMenuGroup title={boardName}>
          <InvMenuItem
            icon={<FaPlus />}
            isDisabled={isSelectedForAutoAdd || autoAssignBoardOnClick}
            onClick={handleSetAutoAdd}
          >
            {t('boards.menuItemAutoAdd')}
          </InvMenuItem>
          {isBulkDownloadEnabled && (
            <InvMenuItem
              icon={<FaDownload />}
              onClickCapture={handleBulkDownload}
            >
              {t('boards.downloadBoard')}
            </InvMenuItem>
          )}
          {board && (
            <GalleryBoardContextMenuItems
              board={board}
              setBoardToDelete={setBoardToDelete}
            />
          )}
        </InvMenuGroup>
      </InvMenuList>
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
    ]
  );

  return (
    <InvContextMenu renderMenu={renderMenuFunc}>{children}</InvContextMenu>
  );
};

export default memo(BoardContextMenu);
