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
import { FaPlus } from 'react-icons/fa';
import { useBoardName } from 'services/api/hooks/useBoardName';
import { BoardDTO } from 'services/api/types';
import { menuListMotionProps } from 'theme/components/menu';
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
              <GalleryBoardContextMenuItems
                board={board}
                setBoardToDelete={setBoardToDelete}
              />
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
