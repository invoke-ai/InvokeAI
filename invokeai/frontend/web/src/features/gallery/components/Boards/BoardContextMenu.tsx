import { MenuGroup, MenuItem, MenuList } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ContextMenu, ContextMenuProps } from 'chakra-ui-contextmenu';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { FaPlus } from 'react-icons/fa';
import { useBoardName } from 'services/api/hooks/useBoardName';
import { BoardDTO } from 'services/api/types';
import { menuListMotionProps } from 'theme/components/menu';
import GalleryBoardContextMenuItems from './GalleryBoardContextMenuItems';
import NoBoardContextMenuItems from './NoBoardContextMenuItems';

type Props = {
  board?: BoardDTO;
  board_id?: string;
  children: ContextMenuProps<HTMLDivElement>['children'];
  setBoardToDelete?: (board?: BoardDTO) => void;
};

const BoardContextMenu = memo(
  ({ board, board_id, setBoardToDelete, children }: Props) => {
    const dispatch = useAppDispatch();

    const selector = useMemo(
      () =>
        createSelector(stateSelector, ({ gallery }) => {
          const isAutoAdd = gallery.autoAddBoardId === board_id;
          return { isAutoAdd };
        }),
      [board_id]
    );

    const { isAutoAdd } = useAppSelector(selector);
    const boardName = useBoardName(board_id);

    const handleSetAutoAdd = useCallback(() => {
      dispatch(autoAddBoardIdChanged(board_id));
    }, [board_id, dispatch]);

    const skipEvent = useCallback((e: MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
    }, []);

    return (
      <ContextMenu<HTMLDivElement>
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
                isDisabled={isAutoAdd}
                onClick={handleSetAutoAdd}
              >
                Auto-add to this Board
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
      </ContextMenu>
    );
  }
);

BoardContextMenu.displayName = 'HoverableBoard';

export default BoardContextMenu;
