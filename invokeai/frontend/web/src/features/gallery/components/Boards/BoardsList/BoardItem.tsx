import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Icon, Image, Text, Tooltip } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { AddImageToBoardDndTargetData, RemoveImageFromBoardDndTargetData } from 'features/dnd/dnd';
import { addImageToBoardDndTarget, removeImageFromBoardDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { AutoAddIndicator } from 'features/gallery/components/Boards/AutoAddIndicator';
import BoardContextMenu from 'features/gallery/components/Boards/BoardContextMenu';
import { BoardEditableTitle } from 'features/gallery/components/Boards/BoardsList/BoardEditableTitle';
import { BoardTooltip } from 'features/gallery/components/Boards/BoardsList/BoardTooltip';
import NoBoardBoardContextMenu from 'features/gallery/components/Boards/NoBoardBoardContextMenu';
import {
  selectAutoAddBoardId,
  selectAutoAssignBoardOnClick,
  selectSelectedBoardId,
} from 'features/gallery/store/gallerySelectors';
import { autoAddBoardIdChanged, boardIdSelected } from 'features/gallery/store/gallerySlice';
import type { ReactNode, RefObject } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArchiveBold, PiImageSquare } from 'react-icons/pi';
import { useGetBoardAssetsTotalQuery, useGetBoardImagesTotalQuery } from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { useBoardName } from 'services/api/hooks/useBoardName';
import type { BoardDTO } from 'services/api/types';

const _hover: SystemStyleObject = {
  bg: 'base.850',
};

type BoardItemProps = {
  /** Pass a BoardDTO for a regular board, or null for the "Uncategorized" board. */
  board: BoardDTO | null;
  isSelected: boolean;
  isCollapsed?: boolean;
}

const BoardItem = ({ board, isSelected, isCollapsed = false }: BoardItemProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const autoAssignBoardOnClick = useAppSelector(selectAutoAssignBoardOnClick);
  const selectedBoardId = useAppSelector(selectSelectedBoardId);

  const boardId = board?.board_id ?? 'none';

  const onClick = useCallback(() => {
    if (board) {
      if (selectedBoardId !== board.board_id) {
        dispatch(boardIdSelected({ boardId: board.board_id }));
      }
      if (autoAssignBoardOnClick && autoAddBoardId !== board.board_id) {
        dispatch(autoAddBoardIdChanged(board.board_id));
      }
    } else {
      dispatch(boardIdSelected({ boardId: 'none' }));
      if (autoAssignBoardOnClick) {
        dispatch(autoAddBoardIdChanged('none'));
      }
    }
  }, [board, selectedBoardId, autoAssignBoardOnClick, autoAddBoardId, dispatch]);

  const dndTargetData = useMemo<AddImageToBoardDndTargetData | RemoveImageFromBoardDndTargetData>(() => {
    if (board) {
      return addImageToBoardDndTarget.getData({ boardId: board.board_id });
    }
    return removeImageFromBoardDndTarget.getData();
  }, [board]);

  const dndTarget = board ? addImageToBoardDndTarget : removeImageFromBoardDndTarget;

  // For the uncategorized board (board === null), counts must be fetched via separate queries.
  // For regular boards, counts are available directly on the BoardDTO.
  const { imagesTotal: noBoardImagesTotal } = useGetBoardImagesTotalQuery(board ? skipToken : 'none', {
    selectFromResult: ({ data }) => ({ imagesTotal: data?.total ?? 0 }),
  });
  const { assetsTotal: noBoardAssetsTotal } = useGetBoardAssetsTotalQuery(board ? skipToken : 'none', {
    selectFromResult: ({ data }) => ({ assetsTotal: data?.total ?? 0 }),
  });

  const boardCounts = useMemo(
    () => ({
      image_count: board ? board.image_count : noBoardImagesTotal,
      asset_count: board ? board.asset_count : noBoardAssetsTotal,
    }),
    [board, noBoardImagesTotal, noBoardAssetsTotal]
  );

  const contextMenuContent = (ref: RefObject<HTMLDivElement>, tooltipLabel: ReactNode, innerContent: ReactNode) => (
    <Tooltip label={tooltipLabel} openDelay={150} placement="right" closeOnScroll>
      <Flex
        ref={ref}
        onClick={onClick}
        alignItems="center"
        borderRadius="base"
        cursor="pointer"
        p={1}
        gap={2}
        bg={isSelected ? 'base.850' : undefined}
        _hover={_hover}
        w="full"
        h="full"
      >
        {innerContent}
      </Flex>
    </Tooltip>
  );

  const titleContent = board ? (
    <Flex flex={1}>
      <BoardEditableTitle board={board} isSelected={isSelected} />
    </Flex>
  ) : (
    <NoBoardTitle isSelected={isSelected} />
  );

  const countsContent = (
    <Flex justifyContent="flex-end" flexShrink={0} pe={3}>
      <Text variant="subtext">
        {boardCounts.image_count} | {boardCounts.asset_count}
      </Text>
    </Flex>
  );

  const tooltipLabel = <BoardTooltip board={board} boardCounts={boardCounts} />;

  const expandedInnerContent = (
    <>
      <BoardThumbnail board={board} />
      {titleContent}
      {autoAddBoardId === boardId && <AutoAddIndicator />}
      {board?.archived && <Icon as={PiArchiveBold} fill="base.300" />}
      {countsContent}
    </>
  );

  const collapsedInnerContent = (
    <>
      <BoardThumbnail board={board} />
    </>
  )

  return (
    <Box position="relative" w="full" h={12}>
      {board ? (
        <BoardContextMenu board={board}>
          {(ref) => contextMenuContent(ref, tooltipLabel, isCollapsed ? collapsedInnerContent : expandedInnerContent)}
        </BoardContextMenu>
      ) : (
        <NoBoardBoardContextMenu>
          {(ref) => contextMenuContent(ref, tooltipLabel, isCollapsed ? collapsedInnerContent : expandedInnerContent)}
        </NoBoardBoardContextMenu>
      )}
      <DndDropTarget dndTarget={dndTarget} dndTargetData={dndTargetData} label={t('gallery.move')} />
    </Box>
  );
};

export default memo(BoardItem);

/**
 * Thumbnail for a board item. Shows the cover image for regular boards,
 * or the Invoke logo SVG for the "Uncategorized" board.
 */
const BoardThumbnail = memo(({ board }: { board: BoardDTO | null }) => {
  const { currentData: coverImage } = useGetImageDTOQuery(board?.cover_image_name ?? skipToken);

  if (board && coverImage) {
    return (
      <Image
        src={coverImage.thumbnail_url}
        draggable={false}
        objectFit="cover"
        w={10}
        h={10}
        borderRadius="base"
        flexShrink={0}
      />
    );
  }

  if (board) {
    return (
      <Flex w={10} h={10} justifyContent="center" alignItems="center">
        <Icon boxSize={10} as={PiImageSquare} opacity={0.7} color="base.500" />
      </Flex>
    );
  }

  // Uncategorized board - show Invoke logo
  return (
    <Flex w={10} h={10} justifyContent="center" alignItems="center">
      <Icon boxSize={8} opacity={1} stroke="base.500" viewBox="0 0 66 66" fill="none">
        <path d="M43.9137 16H63.1211V3H3.12109V16H22.3285L43.9137 50H63.1211V63H3.12109V50H22.3285" strokeWidth="5" />
      </Icon>
    </Flex>
  );
});

BoardThumbnail.displayName = 'BoardThumbnail';

/**
 * Static title for the "Uncategorized" board (not editable).
 */
const NoBoardTitle = memo(({ isSelected }: { isSelected: boolean }) => {
  const boardName = useBoardName('none');
  return (
    <Text fontSize="sm" color={isSelected ? 'base.100' : 'base.300'} fontWeight="semibold" noOfLines={1} flexGrow={1}>
      {boardName}
    </Text>
  );
});

NoBoardTitle.displayName = 'NoBoardTitle';
