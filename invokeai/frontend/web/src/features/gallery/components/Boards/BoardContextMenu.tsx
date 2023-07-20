import { Box, MenuItem, MenuList } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { ContextMenu, ContextMenuProps } from 'chakra-ui-contextmenu';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { FaFolder } from 'react-icons/fa';
import { BoardDTO } from 'services/api/types';
import { menuListMotionProps } from 'theme/components/menu';
import GalleryBoardContextMenuItems from './GalleryBoardContextMenuItems';
import SystemBoardContextMenuItems from './SystemBoardContextMenuItems';

type Props = {
  board?: BoardDTO;
  board_id: string;
  children: ContextMenuProps<HTMLDivElement>['children'];
  setBoardToDelete?: (board?: BoardDTO) => void;
};

const BoardContextMenu = memo(
  ({ board, board_id, setBoardToDelete, children }: Props) => {
    const dispatch = useAppDispatch();
    const handleSelectBoard = useCallback(() => {
      dispatch(boardIdSelected(board?.board_id ?? board_id));
    }, [board?.board_id, board_id, dispatch]);
    return (
      <Box sx={{ touchAction: 'none', height: 'full' }}>
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
      </Box>
    );
  }
);

BoardContextMenu.displayName = 'HoverableBoard';

export default BoardContextMenu;
