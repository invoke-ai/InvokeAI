import { Box, Flex, Image, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import SelectionOverlay from 'common/components/SelectionOverlay';
import type { RemoveFromBoardDropData } from 'features/dnd/types';
import AutoAddIcon from 'features/gallery/components/Boards/AutoAddIcon';
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
  const autoAddBoardId = useAppSelector((s) => s.gallery.autoAddBoardId);
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
    <Box w="full" h="full" userSelect="none">
      <Flex
        onMouseOver={handleMouseOver}
        onMouseOut={handleMouseOut}
        position="relative"
        justifyContent="center"
        alignItems="center"
        aspectRatio="1/1"
        borderRadius="base"
        w="full"
        h="full"
      >
        <NoBoardBoardContextMenu>
          {(ref) => (
            <Tooltip label={<BoardTotalsTooltip board_id="none" isArchived={false} />} openDelay={1000}>
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
                <Flex w="full" h="full" justifyContent="center" alignItems="center">
                  <Image
                    src={InvokeLogoSVG}
                    alt="invoke-ai-logo"
                    opacity={0.7}
                    mixBlendMode="overlay"
                    mt={-6}
                    w={16}
                    h={16}
                    minW={16}
                    minH={16}
                    userSelect="none"
                  />
                </Flex>
                {autoAddBoardId === 'none' && <AutoAddIcon />}
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
                  fontWeight={isSelected ? 'bold' : 'normal'}
                >
                  {boardName}
                </Flex>
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
