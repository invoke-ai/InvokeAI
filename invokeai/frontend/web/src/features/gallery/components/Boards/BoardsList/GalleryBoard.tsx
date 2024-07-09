import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  Icon,
  Image,
  Text,
  Tooltip,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import type { AddToBoardDropData } from 'features/dnd/types';
import BoardContextMenu from 'features/gallery/components/Boards/BoardContextMenu';
import { BoardTotalsTooltip } from 'features/gallery/components/Boards/BoardsList/BoardTotalsTooltip';
import { autoAddBoardIdChanged, boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArchiveBold, PiImageSquare } from 'react-icons/pi';
import { useUpdateBoardMutation } from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

const editableInputStyles: SystemStyleObject = {
  p: 0,
  fontSize: 'md',
  w: '100%',
  _focusVisible: {
    p: 0,
  },
};

const _hover: SystemStyleObject = {
  bg: 'base.800',
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
  const editingDisclosure = useDisclosure();
  const [localBoardName, setLocalBoardName] = useState(board.board_name);

  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected({ boardId: board.board_id }));
    if (autoAssignBoardOnClick) {
      dispatch(autoAddBoardIdChanged(board.board_id));
    }
  }, [dispatch, board.board_id, autoAssignBoardOnClick]);

  const [updateBoard, { isLoading: isUpdateBoardLoading }] = useUpdateBoardMutation();

  const droppableData: AddToBoardDropData = useMemo(
    () => ({
      id: board.board_id,
      actionType: 'ADD_TO_BOARD',
      context: { boardId: board.board_id },
    }),
    [board.board_id]
  );

  const handleSubmit = useCallback(
    async (newBoardName: string) => {
      if (!newBoardName.trim()) {
        // empty strings are not allowed
        setLocalBoardName(board.board_name);
      } else if (newBoardName === board.board_name) {
        // don't updated the board name if it hasn't changed
      } else {
        try {
          const { board_name } = await updateBoard({
            board_id: board.board_id,
            changes: { board_name: newBoardName },
          }).unwrap();

          // update local state
          setLocalBoardName(board_name);
        } catch {
          // revert on error
          setLocalBoardName(board.board_name);
        }
      }
      editingDisclosure.onClose();
    },
    [board.board_id, board.board_name, editingDisclosure, updateBoard]
  );

  const handleChange = useCallback((newBoardName: string) => {
    setLocalBoardName(newBoardName);
  }, []);

  return (
    <BoardContextMenu board={board} setBoardToDelete={setBoardToDelete}>
      {(ref) => (
        <Tooltip
          label={<BoardTotalsTooltip board_id={board.board_id} isArchived={Boolean(board.archived)} />}
          openDelay={1000}
        >
          <Flex
            position="relative"
            ref={ref}
            onClick={handleSelectBoard}
            w="full"
            alignItems="center"
            borderRadius="base"
            cursor="pointer"
            py={1}
            px={2}
            gap={2}
            bg={isSelected ? 'base.800' : undefined}
            _hover={_hover}
          >
            <CoverImage board={board} />
            <Editable
              as={Flex}
              alignItems="center"
              gap={4}
              flexGrow={1}
              onEdit={editingDisclosure.onOpen}
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
                w="fit-content"
                wordBreak="break-all"
                color={isSelected ? 'base.100' : 'base.400'}
                fontWeight={isSelected ? 'semibold' : 'normal'}
              />
              <EditableInput sx={editableInputStyles} />
            </Editable>
            {board.archived && !editingDisclosure.isOpen && (
              <Icon
                as={PiArchiveBold}
                fill="base.300"
                filter="drop-shadow(0px 0px 0.1rem var(--invoke-colors-base-800))"
              />
            )}
            <IAIDroppable data={droppableData} dropLabel={<Text fontSize="md">{t('unifiedCanvas.move')}</Text>} />
          </Flex>
        </Tooltip>
      )}
    </BoardContextMenu>
  );
};

export default memo(GalleryBoard);

const CoverImage = ({ board }: { board: BoardDTO }) => {
  const { currentData: coverImage } = useGetImageDTOQuery(board.cover_image_name ?? skipToken);

  if (coverImage) {
    return (
      <Image
        src={coverImage.thumbnail_url}
        draggable={false}
        objectFit="cover"
        w={8}
        h={8}
        borderRadius="base"
        borderBottomRadius="lg"
      />
    );
  }

  return (
    <Flex w={8} h={8} justifyContent="center" alignItems="center">
      <Icon boxSize={8} as={PiImageSquare} opacity={0.7} color="base.500" />
    </Flex>
  );
};
