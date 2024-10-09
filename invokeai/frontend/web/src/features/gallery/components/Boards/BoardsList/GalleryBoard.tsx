import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Icon, Image, Text, Tooltip } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import type { AddToBoardDropData } from 'features/dnd/types';
import { AutoAddBadge } from 'features/gallery/components/Boards/AutoAddBadge';
import BoardContextMenu from 'features/gallery/components/Boards/BoardContextMenu';
import { BoardEditableTitle } from 'features/gallery/components/Boards/BoardsList/BoardEditableTitle';
import { BoardTooltip } from 'features/gallery/components/Boards/BoardsList/BoardTooltip';
import {
  selectAutoAddBoardId,
  selectAutoAssignBoardOnClick,
  selectSelectedBoardId,
} from 'features/gallery/store/gallerySelectors';
import { autoAddBoardIdChanged, boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArchiveBold, PiImageSquare } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

const _hover: SystemStyleObject = {
  bg: 'base.850',
};

interface GalleryBoardProps {
  board: BoardDTO;
  isSelected: boolean;
}

const GalleryBoard = ({ board, isSelected }: GalleryBoardProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const autoAssignBoardOnClick = useAppSelector(selectAutoAssignBoardOnClick);
  const selectedBoardId = useAppSelector(selectSelectedBoardId);

  const onClick = useCallback(() => {
    if (selectedBoardId !== board.board_id) {
      dispatch(boardIdSelected({ boardId: board.board_id }));
    }
    if (autoAssignBoardOnClick && autoAddBoardId !== board.board_id) {
      dispatch(autoAddBoardIdChanged(board.board_id));
    }
  }, [selectedBoardId, board.board_id, autoAssignBoardOnClick, autoAddBoardId, dispatch]);

  const droppableData: AddToBoardDropData = useMemo(
    () => ({
      id: board.board_id,
      actionType: 'ADD_TO_BOARD',
      context: { boardId: board.board_id },
    }),
    [board.board_id]
  );

  return (
    <Box position="relative" w="full" h={12}>
      <BoardContextMenu board={board}>
        {(ref) => (
          <Tooltip label={<BoardTooltip board={board} />} openDelay={1000} placement="left" closeOnScroll p={2}>
            <Flex
              ref={ref}
              onClick={onClick}
              alignItems="center"
              borderRadius="base"
              cursor="pointer"
              py={1}
              ps={1}
              pe={4}
              gap={4}
              bg={isSelected ? 'base.850' : undefined}
              _hover={_hover}
              w="full"
              h="full"
            >
              <CoverImage board={board} />
              <Flex w="full">
                <BoardEditableTitle board={board} isSelected={isSelected} />
              </Flex>
              {autoAddBoardId === board.board_id && <AutoAddBadge />}
              {board.archived && <Icon as={PiArchiveBold} fill="base.300" />}
              <Text variant="subtext">{board.image_count}</Text>
            </Flex>
          </Tooltip>
        )}
      </BoardContextMenu>
      <IAIDroppable data={droppableData} dropLabel={t('gallery.move')} />
    </Box>
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
        w={10}
        h={10}
        borderRadius="base"
        borderBottomRadius="lg"
      />
    );
  }

  return (
    <Flex w={10} h={10} justifyContent="center" alignItems="center">
      <Icon boxSize={10} as={PiImageSquare} opacity={0.7} color="base.500" />
    </Flex>
  );
};
