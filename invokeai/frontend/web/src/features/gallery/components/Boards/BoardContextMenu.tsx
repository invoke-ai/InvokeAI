import { MenuItem, MenuList } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { ContextMenu, ContextMenuProps } from 'chakra-ui-contextmenu';
import {
  autoAddBoardIdChanged,
  boardIdSelected,
} from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { FaFolder, FaMinus, FaPlus } from 'react-icons/fa';
import { BoardDTO } from 'services/api/types';
import { menuListMotionProps } from 'theme/components/menu';
import GalleryBoardContextMenuItems from './GalleryBoardContextMenuItems';
import SystemBoardContextMenuItems from './SystemBoardContextMenuItems';

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
        createSelector(
          stateSelector,
          ({ gallery }) => {
            const isSelectedForAutoAdd = board_id === gallery.autoAddBoardId;

            return { isSelectedForAutoAdd };
          },
          defaultSelectorOptions
        ),
      [board_id]
    );

    const { isSelectedForAutoAdd } = useAppSelector(selector);

    const handleSelectBoard = useCallback(() => {
      dispatch(boardIdSelected(board?.board_id ?? board_id));
    }, [board?.board_id, board_id, dispatch]);

    const handleAutoAdd = useCallback(() => {
      dispatch(autoAddBoardIdChanged(board_id));
    }, [board_id, dispatch]);

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
          >
            <MenuItem icon={<FaFolder />} onClickCapture={handleSelectBoard}>
              Select Board
            </MenuItem>
            <MenuItem
              icon={isSelectedForAutoAdd ? <FaMinus /> : <FaPlus />}
              onClickCapture={handleAutoAdd}
            >
              {isSelectedForAutoAdd
                ? 'Disable Auto-Add'
                : 'Auto-Add to this Board'}
            </MenuItem>
            {!board && <SystemBoardContextMenuItems board_id={board_id} />}
            {board && (
              <GalleryBoardContextMenuItems
                board={board}
                setBoardToDelete={setBoardToDelete}
              />
            )}
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
