import { MenuGroup, MenuItem, MenuList } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import {
  IAIContextMenu,
  IAIContextMenuProps,
} from 'common/components/IAIContextMenu';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { BoardId } from 'features/gallery/store/types';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaDownload, FaPlus } from 'react-icons/fa';
import { useBoardName } from 'services/api/hooks/useBoardName';
import { BoardDTO } from 'services/api/types';
import { menuListMotionProps } from 'theme/components/menu';
import GalleryBoardContextMenuItems from './GalleryBoardContextMenuItems';
import NoBoardContextMenuItems from './NoBoardContextMenuItems';
import { useFeatureStatus } from '../../../system/hooks/useFeatureStatus';
import { useBulkDownloadImagesMutation } from '../../../../services/api/endpoints/images';
import { addToast } from '../../../system/store/systemSlice';

type Props = {
  board?: BoardDTO;
  board_id: BoardId;
  children: IAIContextMenuProps<HTMLDivElement>['children'];
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
      createSelector(
        stateSelector,
        ({ gallery }) => {
          const isAutoAdd = gallery.autoAddBoardId === board_id;
          const autoAssignBoardOnClick = gallery.autoAssignBoardOnClick;
          return { isAutoAdd, autoAssignBoardOnClick };
        },
        defaultSelectorOptions
      ),
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
          ...(response.response ? { description: response.response } : {}),
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
      <MenuList
        sx={{ visibility: 'visible !important' }}
        motionProps={menuListMotionProps}
        onContextMenu={skipEvent}
      >
        <MenuGroup title={boardName}>
          <MenuItem
            icon={<FaPlus />}
            isDisabled={isAutoAdd || autoAssignBoardOnClick}
            onClick={handleSetAutoAdd}
          >
            {t('boards.menuItemAutoAdd')}
          </MenuItem>
          {isBulkDownloadEnabled && (
            <MenuItem icon={<FaDownload />} onClickCapture={handleBulkDownload}>
              {t('boards.downloadBoard')}
            </MenuItem>
          )}
          {!board && <NoBoardContextMenuItems />}
          {board && (
            <GalleryBoardContextMenuItems
              board={board}
              setBoardToDelete={setBoardToDelete}
            />
          )}
        </MenuGroup>
      </MenuList>
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
    <IAIContextMenu<HTMLDivElement>
      menuProps={{ size: 'sm', isLazy: true }}
      menuButtonProps={{
        bg: 'transparent',
        _hover: { bg: 'transparent' },
      }}
      renderMenu={renderMenuFunc}
    >
      {children}
    </IAIContextMenu>
  );
};

export default memo(BoardContextMenu);
