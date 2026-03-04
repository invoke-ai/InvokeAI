import { Flex, Icon, Image, Tooltip } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { selectListBoardsQueryArgs, selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo, useCallback, useMemo } from 'react';
import { PiImageSquare } from 'react-icons/pi';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { useBoardName } from 'services/api/hooks/useBoardName';

const OVERLAY_SCROLLBARS_STYLES = { height: '100%', width: '100%' };

const CollapsedBoardThumbnail = memo(
  ({
    boardId,
    coverImageName,
    isSelected,
    onSelectBoard,
    label,
  }: {
    boardId: string;
    coverImageName?: string;
    isSelected: boolean;
    onSelectBoard: (boardId: string) => void;
    label: string;
  }) => {
    const { currentData: coverImage } = useGetImageDTOQuery(coverImageName ?? skipToken);
    const onClick = useCallback(() => onSelectBoard(boardId), [boardId, onSelectBoard]);

    return (
      <Tooltip label={label} openDelay={150} placement="right" closeOnScroll>
        <Flex
          as="button"
          onClick={onClick}
          w={12}
          h={12}
          borderRadius="base"
          alignItems="center"
          justifyContent="center"
          bg={isSelected ? 'base.750' : undefined}
          borderWidth={1.5}
          borderColor={isSelected ? 'invokeBlue.400' : 'base.750'}
          _hover={{ bg: 'base.850' }}
          overflow="hidden"
        >
          {coverImage ? (
            <Image src={coverImage.thumbnail_url} draggable={false} objectFit="cover" w="full" h="full" />
          ) : (
            <Icon as={PiImageSquare} color="base.400" boxSize={6} />
          )}
        </Flex>
      </Tooltip>
    );
  }
);

CollapsedBoardThumbnail.displayName = 'CollapsedBoardThumbnail';

export const CollapsedBoardsList = memo(() => {
  const dispatch = useAppDispatch();
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const queryArgs = useAppSelector(selectListBoardsQueryArgs);
  const { data: boards } = useListAllBoardsQuery(queryArgs);
  const noBoardName = useBoardName('none');
  const items = useMemo(
    () =>
      [{ boardId: 'none', label: noBoardName, coverImageName: undefined as string | undefined }].concat(
        (boards ?? []).map((board) => ({
          boardId: board.board_id,
          label: board.board_name,
          coverImageName: board.cover_image_name,
        }))
      ),
    [boards, noBoardName]
  );

  const onSelectBoard = useCallback((boardId: string) => dispatch(boardIdSelected({ boardId })), [dispatch]);

  return (
    <OverlayScrollbarsComponent style={OVERLAY_SCROLLBARS_STYLES} options={overlayScrollbarsParams.options}>
      <Flex direction="column" alignItems="center" gap={1} px={0.5} py={1}>
        {items.map((board) => {
          return (
            <CollapsedBoardThumbnail
              key={board.boardId}
              boardId={board.boardId}
              coverImageName={board.coverImageName}
              label={board.label}
              isSelected={selectedBoardId === board.boardId}
              onSelectBoard={onSelectBoard}
            />
          );
        })}
      </Flex>
    </OverlayScrollbarsComponent>
  );
});

CollapsedBoardsList.displayName = 'CollapsedBoardsList';
