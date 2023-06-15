import {
  Box,
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  Icon,
  Image,
  MenuItem,
  MenuList,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { memo, useCallback } from 'react';
import { FaFolder, FaTrash } from 'react-icons/fa';
import { ContextMenu } from 'chakra-ui-contextmenu';
import { BoardDTO } from 'services/api';
import { IAIImageFallback } from 'common/components/IAIImageFallback';
import { boardIdSelected } from 'features/gallery/store/boardSlice';
import { boardDeleted, boardUpdated } from '../../../../services/thunks/board';

interface HoverableBoardProps {
  board: BoardDTO;
  isSelected: boolean;
}

const HoverableBoard = memo(({ board, isSelected }: HoverableBoardProps) => {
  const dispatch = useAppDispatch();

  const { board_name, board_id, cover_image_url } = board;

  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected(board_id));
  }, [board_id, dispatch]);

  const handleDeleteBoard = useCallback(() => {
    dispatch(boardDeleted(board_id));
  }, [board_id, dispatch]);

  const handleUpdateBoardName = (newBoardName: string) => {
    dispatch(
      boardUpdated({
        boardId: board_id,
        requestBody: { board_name: newBoardName },
      })
    );
  };

  return (
    <Box sx={{ touchAction: 'none' }}>
      <ContextMenu<HTMLDivElement>
        menuProps={{ size: 'sm', isLazy: true }}
        renderMenu={() => (
          <MenuList sx={{ visibility: 'visible !important' }}>
            <MenuItem
              sx={{ color: 'error.300' }}
              icon={<FaTrash />}
              onClickCapture={handleDeleteBoard}
            >
              Delete Board
            </MenuItem>
          </MenuList>
        )}
      >
        {(ref) => (
          <Flex
            position="relative"
            key={board_id}
            userSelect="none"
            ref={ref}
            sx={{
              flexDir: 'column',
              justifyContent: 'space-between',
              alignItems: 'center',
              cursor: 'pointer',
              w: 'full',
              h: 'full',
              gap: 1,
            }}
          >
            <Flex
              onClick={handleSelectBoard}
              sx={{
                justifyContent: 'center',
                alignItems: 'center',
                borderWidth: '1px',
                borderRadius: 'base',
                borderColor: isSelected ? 'base.500' : 'base.800',
                w: 'full',
                h: 'full',
                aspectRatio: '1/1',
                overflow: 'hidden',
              }}
            >
              {cover_image_url ? (
                <Image
                  loading="lazy"
                  objectFit="cover"
                  draggable={false}
                  rounded="md"
                  src={cover_image_url}
                  fallback={<IAIImageFallback />}
                  sx={{}}
                />
              ) : (
                <Icon boxSize={8} color="base.700" as={FaFolder} />
              )}
            </Flex>

            <Editable
              defaultValue={board_name}
              submitOnBlur={false}
              onSubmit={(nextValue) => {
                handleUpdateBoardName(nextValue);
              }}
            >
              <EditablePreview
                sx={{ color: 'base.200', fontSize: 'xs', textAlign: 'left' }}
                noOfLines={1}
              />
              <EditableInput
                sx={{
                  color: 'base.200',
                  fontSize: 'xs',
                  textAlign: 'left',
                  borderColor: 'base.500',
                }}
              />
            </Editable>
          </Flex>
        )}
      </ContextMenu>
    </Box>
  );
});

HoverableBoard.displayName = 'HoverableBoard';

export default HoverableBoard;
