import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
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
import type { MouseEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaDownload, FaPlus } from 'react-icons/fa';
import { useBulkDownloadImagesMutation } from 'services/api/endpoints/images';
import { useBoardName } from 'services/api/hooks/useBoardName';
import type { BoardDTO } from 'services/api/types';

import GalleryBoardContextMenuItems from './GalleryBoardContextMenuItems';

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

  const selector = useMemo(
    () =>
      createMemoizedSelector(selectGallerySlice, (gallery) => {
        const isAutoAdd = gallery.autoAddBoardId === board_id;
        const autoAssignBoardOnClick = gallery.autoAssignBoardOnClick;
        return { isAutoAdd, autoAssignBoardOnClick };
      }),
    [board_id]
  );

  const { isAutoAdd, autoAssignBoardOnClick } = useAppSelector(selector);
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

  const skipEvent = useCallback((e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  const renderMenuFunc = useCallback(
    () => (
      <InvMenuList visibility="visible" onContextMenu={skipEvent}>
        <InvMenuGroup title={boardName}>
          <InvMenuItem
            icon={<FaPlus />}
            isDisabled={isAutoAdd || autoAssignBoardOnClick}
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
      isAutoAdd,
      isBulkDownloadEnabled,
      setBoardToDelete,
      skipEvent,
      t,
    ]
  );

  return (
    <InvContextMenu renderMenu={renderMenuFunc}>{children}</InvContextMenu>
  );
};

export default memo(BoardContextMenu);
