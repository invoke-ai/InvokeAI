import {
  Box,
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  MenuItem,
  MenuList,
} from '@chakra-ui/react';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { memo, useCallback } from 'react';
import { FaTrash } from 'react-icons/fa';
import { ContextMenu } from 'chakra-ui-contextmenu';
import { BoardDTO, ImageDTO } from 'services/api';
import { IAIImageFallback } from 'common/components/IAIImageFallback';
import { boardIdSelected } from 'features/gallery/store/boardSlice';
import {
  boardDeleted,
  boardUpdated,
  imageAddedToBoard,
} from '../../../../services/thunks/board';
import { selectImagesAll } from '../../store/imagesSlice';
import IAIDndImage from '../../../../common/components/IAIDndImage';
import { defaultSelectorOptions } from '../../../../app/store/util/defaultMemoizeOptions';
import { createSelector } from '@reduxjs/toolkit';

const selector = createSelector(
  [selectImagesAll],
  (images) => {
    return { images };
  },
  defaultSelectorOptions
);

interface HoverableBoardProps {
  board: BoardDTO;
  isSelected: boolean;
}

const HoverableBoard = memo(({ board, isSelected }: HoverableBoardProps) => {
  const dispatch = useAppDispatch();
  const { images } = useAppSelector(selector);

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

  const handleDrop = useCallback(
    (droppedImage: ImageDTO) => {
      if (droppedImage.board_id === board_id) {
        return;
      }
      dispatch(
        imageAddedToBoard({
          requestBody: {
            board_id,
            image_name: droppedImage.image_name,
          },
        })
      );
    },
    [board_id, dispatch]
  );

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
              <IAIDndImage
                image={cover_image_url ? images[0] : undefined}
                onDrop={handleDrop}
                fallback={<IAIImageFallback sx={{ bg: 'none' }} />}
                isUploadDisabled={true}
              />
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
