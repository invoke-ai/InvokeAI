import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Editable, EditableInput, EditablePreview, Flex, Icon, Image, Text, Tooltip } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import SelectionOverlay from 'common/components/SelectionOverlay';
import type { AddToBoardDropData } from 'features/dnd/types';
import AutoAddIcon from 'features/gallery/components/Boards/AutoAddIcon';
import BoardContextMenu from 'features/gallery/components/Boards/BoardContextMenu';
import { autoAddBoardIdChanged, boardIdSelected, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesSquare } from 'react-icons/pi';
import {
  useGetBoardAssetsTotalQuery,
  useGetBoardImagesTotalQuery,
  useUpdateBoardMutation,
} from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

const editableInputStyles: SystemStyleObject = {
  p: 0,
  _focusVisible: {
    p: 0,
    textAlign: 'center',
  },
};

interface GalleryBoardProps {
  board: BoardDTO;
  isSelected: boolean;
  setBoardToDelete: (board?: BoardDTO) => void;
}

const GalleryBoard = ({ board, isSelected, setBoardToDelete }: GalleryBoardProps) => {
  const dispatch = useAppDispatch();
  const autoAssignBoardOnClick = useAppSelector((s) => s.gallery.autoAssignBoardOnClick);
  const selectIsSelectedForAutoAdd = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => board.board_id === gallery.autoAddBoardId),
    [board.board_id]
  );

  const isSelectedForAutoAdd = useAppSelector(selectIsSelectedForAutoAdd);
  const [isHovered, setIsHovered] = useState(false);
  const handleMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);
  const handleMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const { data: imagesTotal } = useGetBoardImagesTotalQuery(board.board_id);
  const { data: assetsTotal } = useGetBoardAssetsTotalQuery(board.board_id);
  const tooltip = useMemo(() => {
    if (imagesTotal?.total === undefined || assetsTotal?.total === undefined) {
      return undefined;
    }
    return `${imagesTotal.total} image${imagesTotal.total === 1 ? '' : 's'}, ${
      assetsTotal.total
    } asset${assetsTotal.total === 1 ? '' : 's'}`;
  }, [assetsTotal, imagesTotal]);

  const { currentData: coverImage } = useGetImageDTOQuery(board.cover_image_name ?? skipToken);

  const { board_name, board_id } = board;
  const [localBoardName, setLocalBoardName] = useState(board_name);

  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected({ boardId: board_id }));
    if (autoAssignBoardOnClick) {
      dispatch(autoAddBoardIdChanged(board_id));
    }
  }, [board_id, autoAssignBoardOnClick, dispatch]);

  const [updateBoard, { isLoading: isUpdateBoardLoading }] = useUpdateBoardMutation();

  const droppableData: AddToBoardDropData = useMemo(
    () => ({
      id: board_id,
      actionType: 'ADD_TO_BOARD',
      context: { boardId: board_id },
    }),
    [board_id]
  );

  const handleSubmit = useCallback(
    async (newBoardName: string) => {
      // empty strings are not allowed
      if (!newBoardName.trim()) {
        setLocalBoardName(board_name);
        return;
      }

      // don't updated the board name if it hasn't changed
      if (newBoardName === board_name) {
        return;
      }

      try {
        const { board_name } = await updateBoard({
          board_id,
          changes: { board_name: newBoardName },
        }).unwrap();

        // update local state
        setLocalBoardName(board_name);
      } catch {
        // revert on error
        setLocalBoardName(board_name);
      }
    },
    [board_id, board_name, updateBoard]
  );

  const handleChange = useCallback((newBoardName: string) => {
    setLocalBoardName(newBoardName);
  }, []);
  const { t } = useTranslation();
  return (
    <Box w="full" h="full" userSelect="none">
      <Flex
        onMouseOver={handleMouseOver}
        onMouseOut={handleMouseOut}
        position="relative"
        justifyContent="center"
        alignItems="center"
        aspectRatio="1/1"
        w="full"
        h="full"
      >
        <BoardContextMenu board={board} board_id={board_id} setBoardToDelete={setBoardToDelete}>
          {(ref) => (
            <Tooltip label={tooltip} openDelay={1000}>
              <Flex
                ref={ref}
                onClick={handleSelectBoard}
                w="full"
                h="full"
                position="relative"
                justifyContent="center"
                alignItems="center"
                borderRadius="base"
                cursor="pointer"
                bg="base.800"
              >
                {coverImage?.thumbnail_url ? (
                  <Image
                    src={coverImage?.thumbnail_url}
                    draggable={false}
                    objectFit="cover"
                    w="full"
                    h="full"
                    maxH="full"
                    borderRadius="base"
                    borderBottomRadius="lg"
                  />
                ) : (
                  <Flex w="full" h="full" justifyContent="center" alignItems="center">
                    <Icon boxSize={14} as={PiImagesSquare} mt={-6} opacity={0.7} color="base.500" />
                  </Flex>
                )}
                {isSelectedForAutoAdd && <AutoAddIcon />}
                <SelectionOverlay isSelected={isSelected} isHovered={isHovered} />
                <Flex
                  position="absolute"
                  bottom={0}
                  left={0}
                  p={1}
                  justifyContent="center"
                  alignItems="center"
                  w="full"
                  maxW="full"
                  borderBottomRadius="base"
                  bg={isSelected ? 'invokeBlue.400' : 'base.600'}
                  color={isSelected ? 'base.800' : 'base.100'}
                  lineHeight="short"
                  fontSize="xs"
                >
                  <Editable
                    value={localBoardName}
                    isDisabled={isUpdateBoardLoading}
                    submitOnBlur={true}
                    onChange={handleChange}
                    onSubmit={handleSubmit}
                    w="full"
                  >
                    <EditablePreview
                      p={0}
                      fontWeight={isSelected ? 'bold' : 'normal'}
                      textAlign="center"
                      overflow="hidden"
                      textOverflow="ellipsis"
                      noOfLines={1}
                      color="inherit"
                    />
                    <EditableInput sx={editableInputStyles} />
                  </Editable>
                </Flex>

                <IAIDroppable data={droppableData} dropLabel={<Text fontSize="md">{t('unifiedCanvas.move')}</Text>} />
              </Flex>
            </Tooltip>
          )}
        </BoardContextMenu>
      </Flex>
    </Box>
  );
};

export default memo(GalleryBoard);
