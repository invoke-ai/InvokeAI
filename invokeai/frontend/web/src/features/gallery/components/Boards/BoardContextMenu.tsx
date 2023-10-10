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
import { FaPlus, FaDownload } from 'react-icons/fa';
import { useBoardName } from 'services/api/hooks/useBoardName';
import { BoardDTO } from 'services/api/types';
import { menuListMotionProps } from 'theme/components/menu';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import GalleryBoardContextMenuItems from './GalleryBoardContextMenuItems';
import NoBoardContextMenuItems from './NoBoardContextMenuItems';

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

  const handleSetAutoAdd = useCallback(() => {
    dispatch(autoAddBoardIdChanged(board_id));
  }, [board_id, dispatch]);

  const skipEvent = useCallback((e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  const { t } = useTranslation();

  const handleDownloadImages = useCallback(() => {
    const endpoint = `/api/v1/board_images/${board_id}/export-to-zip`;

    fetch(endpoint)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.blob();
      })
      .then((blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${boardName}_Board_images.zip`;
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);

        dispatch(
          addToast(
            makeToast({
              title: 'Notification',
              description: 'Zip file was Made successfully!',
              status: 'success',
            })
          )
        );
      })
      .catch((error) => {
        console.error(
          'There was a problem with the fetch operation:',
          error.message
        );
        dispatch(
          addToast(
            makeToast({
              title: 'Error',
              description: 'Failed to download the zip file.',
              status: 'error',
            })
          )
        );
      });
  }, [board_id, boardName, dispatch]);

  return (
    <IAIContextMenu<HTMLDivElement>
      menuProps={{ size: 'sm', isLazy: true }}
      menuButtonProps={{
        bg: 'transparent',
        _hover: { bg: 'transparent' },
      }}
      renderMenu={() => (
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
            {!board && <NoBoardContextMenuItems />}
            {board && (
              <>
                <MenuItem icon={<FaDownload />} onClick={handleDownloadImages}>
                  {t('boards.menuItemDownloadImages')}
                </MenuItem>
                <GalleryBoardContextMenuItems
                  board={board}
                  setBoardToDelete={setBoardToDelete}
                />
              </>
            )}
          </MenuGroup>
        </MenuList>
      )}
    >
      {children}
    </IAIContextMenu>
  );
};

export default memo(BoardContextMenu);
