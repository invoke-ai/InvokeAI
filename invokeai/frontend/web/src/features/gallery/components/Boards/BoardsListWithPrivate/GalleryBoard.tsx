import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Editable, EditableInput, EditablePreview, Flex, Icon, Image, Text, Tooltip } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import SelectionOverlay from 'common/components/SelectionOverlay';
import type { AddToBoardDropData } from 'features/dnd/types';
import BoardContextMenu from 'features/gallery/components/Boards/BoardContextMenu';
import { BoardTotalsTooltip } from 'features/gallery/components/Boards/BoardsList/BoardTotalsTooltip';
import { autoAddBoardIdChanged, boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArchiveBold, PiImagesSquare } from 'react-icons/pi';
import { useUpdateBoardMutation } from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

const editableInputStyles: SystemStyleObject = {
  p: 0,
  fontSize: 'md',
  w: '100%',
};

const ArchivedIcon = () => {
  return (
    <Box position="absolute" top={1} insetInlineEnd={2} p={0} minW={0}>
      <Icon as={PiArchiveBold} fill="base.300" filter="drop-shadow(0px 0px 0.1rem var(--invoke-colors-base-800))" />
    </Box>
  );
};

interface GalleryBoardProps {
  board: BoardDTO;
  isSelected: boolean;
  setBoardToDelete: (board?: BoardDTO) => void;
}

const GalleryBoard = ({ board, isSelected, setBoardToDelete }: GalleryBoardProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const autoAssignBoardOnClick = useAppSelector((s) => s.gallery.autoAssignBoardOnClick);

  const [isHovered, setIsHovered] = useState(false);
  const handleMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);
  const handleMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const { currentData: coverImage } = useGetImageDTOQuery(board.cover_image_name ?? skipToken);

  const { board_name, board_id } = board;
  const [localBoardName, setLocalBoardName] = useState(board_name);

  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected({ boardId: board_id }));
    if (autoAssignBoardOnClick && !board.archived) {
      dispatch(autoAddBoardIdChanged(board_id));
    }
  }, [board_id, autoAssignBoardOnClick, dispatch, board.archived]);

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

  return (
    <Box w="full" userSelect="none" px="1">
      <Flex
        onMouseOver={handleMouseOver}
        onMouseOut={handleMouseOut}
        position="relative"
        alignItems="center"
        borderRadius="base"
        w="full"
        my="2"
        userSelect="none"
      >
        <BoardContextMenu board={board} setBoardToDelete={setBoardToDelete}>
          {(ref) => (
            <Tooltip
              label={<BoardTotalsTooltip board_id={board.board_id} isArchived={Boolean(board.archived)} />}
              openDelay={1000}
            >
              <Flex
                ref={ref}
                onClick={handleSelectBoard}
                w="full"
                alignItems="center"
                justifyContent="space-between"
                borderRadius="base"
                cursor="pointer"
                gap="6"
                p="1"
              >
                <Flex gap="6">
                {board.archived && <ArchivedIcon />}
                {coverImage?.thumbnail_url ? (
                  <Image
                    src={coverImage?.thumbnail_url}
                    draggable={false}
                    objectFit="cover"
                    w="8"
                    h="8"
                    borderRadius="base"
                    borderBottomRadius="lg"
                  />
                ) : (
                  <Flex w="8" h="8" justifyContent="center" alignItems="center">
                    <Icon boxSize={8} as={PiImagesSquare} opacity={0.7} color="base.500" />
                  </Flex>
                )}

                <SelectionOverlay isSelected={isSelected} isSelectedForCompare={false} isHovered={isHovered} />
                <Flex
                  p={1}
                  justifyContent="center"
                  alignItems="center"
                  color={isSelected ? 'base.100' : 'base.400'}
                  lineHeight="short"
                  fontSize="md"
                >
                  <Editable
                    value={localBoardName}
                    isDisabled={isUpdateBoardLoading}
                    submitOnBlur={true}
                    onChange={handleChange}
                    onSubmit={handleSubmit}
                  >
                    <EditablePreview
                      p={0}
                      fontSize="md"
                      textOverflow="ellipsis"
                      noOfLines={1}
                      color="inherit"
                      w="fit-content"
                    />
                    <EditableInput sx={editableInputStyles} />
                  </Editable>
                </Flex>
                </Flex>
                <Text justifySelf="end" color="base.600">{board.image_count} images</Text>

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
