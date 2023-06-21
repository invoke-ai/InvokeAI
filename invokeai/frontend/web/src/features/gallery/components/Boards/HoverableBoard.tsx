import {
  Box,
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  MenuItem,
  MenuList,
  Text,
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
import { selectImagesAll, selectImagesById } from '../../store/imagesSlice';
import IAIDndImage from '../../../../common/components/IAIDndImage';
import { defaultSelectorOptions } from '../../../../app/store/util/defaultMemoizeOptions';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from '../../../../app/store/store';
import {
  useAddImageToBoardMutation,
  useDeleteBoardMutation,
  useGetImageDTOQuery,
  useUpdateBoardMutation,
} from 'services/apiSlice';
import { skipToken } from '@reduxjs/toolkit/dist/query';

const coverImageSelector = (imageName: string | undefined) =>
  createSelector(
    [(state: RootState) => state],
    (state) => {
      const coverImage = imageName
        ? selectImagesById(state, imageName)
        : undefined;

      return {
        coverImage,
      };
    },
    defaultSelectorOptions
  );

interface HoverableBoardProps {
  board: BoardDTO;
  isSelected: boolean;
}

const HoverableBoard = memo(({ board, isSelected }: HoverableBoardProps) => {
  const dispatch = useAppDispatch();

  const { data: coverImage } = useGetImageDTOQuery(
    board.cover_image_name ?? skipToken
  );

  const { board_name, board_id } = board;

  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected(board_id));
  }, [board_id, dispatch]);

  const [updateBoard, { isLoading: isUpdateBoardLoading }] =
    useUpdateBoardMutation();

  const [deleteBoard, { isLoading: isDeleteBoardLoading }] =
    useDeleteBoardMutation();

  const [addImageToBoard, { isLoading: isAddImageToBoardLoading }] =
    useAddImageToBoardMutation();

  const handleUpdateBoardName = (newBoardName: string) => {
    updateBoard({ board_id, changes: { board_name: newBoardName } });
  };

  const handleDeleteBoard = useCallback(() => {
    deleteBoard(board_id);
  }, [board_id, deleteBoard]);

  const handleDrop = useCallback(
    (droppedImage: ImageDTO) => {
      if (droppedImage.board_id === board_id) {
        return;
      }
      addImageToBoard({ board_id, image_name: droppedImage.image_name });
    },
    [addImageToBoard, board_id]
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
                image={
                  board.cover_image_name && coverImage ? coverImage : undefined
                }
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
            <Flex
              sx={{
                justifyContent: 'center',
                alignItems: 'center',
                pos: 'absolute',
                color: 'base.900',
                bg: 'accent.300',
                borderRadius: 'full',
                w: 4,
                h: 4,
                right: -1,
                top: -1,
              }}
            >
              <Text fontSize="2xs">{board.image_count}</Text>
            </Flex>
          </Flex>
        )}
      </ContextMenu>
    </Box>
  );
});

HoverableBoard.displayName = 'HoverableBoard';

export default HoverableBoard;
