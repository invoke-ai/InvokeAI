import { Box, Flex, Image, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import SelectionOverlay from 'common/components/SelectionOverlay';
import type { RemoveFromBoardDropData } from 'features/dnd/types';
import { BoardTotalsTooltip } from 'features/gallery/components/Boards/BoardsList/BoardTotalsTooltip';
import NoBoardBoardContextMenu from 'features/gallery/components/Boards/NoBoardBoardContextMenu';
import { autoAddBoardIdChanged, boardIdSelected } from 'features/gallery/store/gallerySlice';
import InvokeLogoSVG from 'public/assets/images/invoke-symbol-wht-lrg.svg';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useBoardName } from 'services/api/hooks/useBoardName';

interface Props {
  isSelected: boolean;
}

const NoBoardBoard = memo(({ isSelected }: Props) => {
  const dispatch = useAppDispatch();
  const autoAssignBoardOnClick = useAppSelector((s) => s.gallery.autoAssignBoardOnClick);
  const boardName = useBoardName('none');
  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected({ boardId: 'none' }));
    if (autoAssignBoardOnClick) {
      dispatch(autoAddBoardIdChanged('none'));
    }
  }, [dispatch, autoAssignBoardOnClick]);
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const handleMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const droppableData: RemoveFromBoardDropData = useMemo(
    () => ({
      id: 'no_board',
      actionType: 'REMOVE_FROM_BOARD',
    }),
    []
  );
  const { t } = useTranslation();
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
        <NoBoardBoardContextMenu>
          {(ref) => (
            <Tooltip label={<BoardTotalsTooltip board_id="none" isArchived={false} />} openDelay={1000}>
              <Flex
                ref={ref}
                onClick={handleSelectBoard}
                w="full"
                alignItems="center"
                borderRadius="base"
                cursor="pointer"
                gap="6"
                p="1"
              >
                <Image
                  src={InvokeLogoSVG}
                  alt="invoke-ai-logo"
                  opacity={0.7}
                  mixBlendMode="overlay"
                  userSelect="none"
                  height="6"
                  width="6"
                />
                <Text fontSize="md" color={isSelected ? 'base.100' : 'base.400'}>
                  {boardName}
                </Text>
                <SelectionOverlay isSelected={isSelected} isSelectedForCompare={false} isHovered={isHovered} />
                <IAIDroppable data={droppableData} dropLabel={<Text fontSize="md">{t('unifiedCanvas.move')}</Text>} />
              </Flex>
            </Tooltip>
          )}
        </NoBoardBoardContextMenu>
      </Flex>
    </Box>
  );
});

NoBoardBoard.displayName = 'HoverableBoard';

export default memo(NoBoardBoard);
