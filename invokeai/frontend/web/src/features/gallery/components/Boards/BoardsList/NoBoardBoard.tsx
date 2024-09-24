import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Icon, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import type { RemoveFromBoardDropData } from 'features/dnd/types';
import { AutoAddBadge } from 'features/gallery/components/Boards/AutoAddBadge';
import { BoardTooltip } from 'features/gallery/components/Boards/BoardsList/BoardTooltip';
import NoBoardBoardContextMenu from 'features/gallery/components/Boards/NoBoardBoardContextMenu';
import {
  selectAutoAddBoardId,
  selectAutoAssignBoardOnClick,
  selectBoardSearchText,
} from 'features/gallery/store/gallerySelectors';
import { autoAddBoardIdChanged, boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetUncategorizedImageCountsQuery } from 'services/api/endpoints/boards';
import { useBoardName } from 'services/api/hooks/useBoardName';

interface Props {
  isSelected: boolean;
}

const _hover: SystemStyleObject = {
  bg: 'base.850',
};

const NoBoardBoard = memo(({ isSelected }: Props) => {
  const dispatch = useAppDispatch();
  const { data } = useGetUncategorizedImageCountsQuery();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const autoAssignBoardOnClick = useAppSelector(selectAutoAssignBoardOnClick);
  const boardSearchText = useAppSelector(selectBoardSearchText);
  const boardName = useBoardName('none');
  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected({ boardId: 'none' }));
    if (autoAssignBoardOnClick) {
      dispatch(autoAddBoardIdChanged('none'));
    }
  }, [dispatch, autoAssignBoardOnClick]);

  const droppableData: RemoveFromBoardDropData = useMemo(
    () => ({
      id: 'no_board',
      actionType: 'REMOVE_FROM_BOARD',
    }),
    []
  );

  const { t } = useTranslation();

  if (boardSearchText.length) {
    return null;
  }

  return (
    <NoBoardBoardContextMenu>
      {(ref) => (
        <Tooltip
          label={
            <BoardTooltip imageCount={data?.image_count ?? 0} assetCount={data?.asset_count ?? 0} isArchived={false} />
          }
          openDelay={500}
          placement="left"
          closeOnScroll
        >
          <Flex
            position="relative"
            ref={ref}
            onPointerUp={handleSelectBoard}
            w="full"
            alignItems="center"
            borderRadius="base"
            cursor="pointer"
            py={1}
            ps={1}
            pe={4}
            gap={4}
            bg={isSelected ? 'base.850' : undefined}
            _hover={_hover}
            h={12}
          >
            <Flex w="10" justifyContent="space-around">
              {/* iconified from public/assets/images/invoke-symbol-wht-lrg.svg */}
              <Icon boxSize={8} opacity={1} stroke="base.500" viewBox="0 0 66 66" fill="none">
                <path
                  d="M43.9137 16H63.1211V3H3.12109V16H22.3285L43.9137 50H63.1211V63H3.12109V50H22.3285"
                  strokeWidth="5"
                />
              </Icon>
            </Flex>

            <Text
              fontSize="sm"
              color={isSelected ? 'base.100' : 'base.300'}
              fontWeight="semibold"
              noOfLines={1}
              flexGrow={1}
            >
              {boardName}
            </Text>
            {autoAddBoardId === 'none' && <AutoAddBadge />}
            <Text variant="subtext">{(data?.image_count ?? 0) + (data?.asset_count ?? 0)}</Text>
            <IAIDroppable data={droppableData} dropLabel={t('gallery.move')} />
          </Flex>
        </Tooltip>
      )}
    </NoBoardBoardContextMenu>
  );
});

NoBoardBoard.displayName = 'HoverableBoard';

export default memo(NoBoardBoard);
