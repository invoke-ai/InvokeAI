import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Icon, Image, Text, Tooltip } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { PiCalendarBold, PiImageSquare } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { VirtualSubBoard } from 'services/api/endpoints/virtual_boards';

const _hover: SystemStyleObject = {
  bg: 'base.850',
};

interface VirtualBoardItemProps {
  board: VirtualSubBoard;
}

const VirtualBoardItem = ({ board }: VirtualBoardItemProps) => {
  const dispatch = useAppDispatch();
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const isSelected = selectedBoardId === board.virtual_board_id;

  const onClick = useCallback(() => {
    if (selectedBoardId !== board.virtual_board_id) {
      dispatch(boardIdSelected({ boardId: board.virtual_board_id }));
    }
  }, [selectedBoardId, board.virtual_board_id, dispatch]);

  return (
    <Box position="relative" w="full" h={12}>
      <Tooltip
        label={`${board.date} — ${board.image_count} images, ${board.asset_count} assets`}
        openDelay={1000}
        placement="right"
        closeOnScroll
        p={2}
      >
        <Flex
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
          <CoverImage coverImageName={board.cover_image_name} />
          <Flex flex={1} direction="column" minW={0}>
            <Text fontSize="sm" noOfLines={1} fontWeight={isSelected ? 'bold' : 'normal'}>
              {board.board_name}
            </Text>
          </Flex>
          <Icon as={PiCalendarBold} fill="base.500" boxSize={4} />
          <Flex justifyContent="flex-end">
            <Text variant="subtext">
              {board.image_count} | {board.asset_count}
            </Text>
          </Flex>
        </Flex>
      </Tooltip>
    </Box>
  );
};

export default memo(VirtualBoardItem);

const CoverImage = ({ coverImageName }: { coverImageName: string | null }) => {
  const { currentData: coverImage } = useGetImageDTOQuery(coverImageName ?? skipToken);

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
