import { Box, Flex, Image, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import InvokeAILogoImage from 'assets/images/logo.png';
import IAIDroppable from 'common/components/IAIDroppable';
import SelectionOverlay from 'common/components/SelectionOverlay';
import { RemoveFromBoardDropData } from 'features/dnd/types';
import {
  autoAddBoardIdChanged,
  boardIdSelected,
} from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo, useState } from 'react';
import { useBoardName } from 'services/api/hooks/useBoardName';
import AutoAddIcon from '../AutoAddIcon';
import BoardContextMenu from '../BoardContextMenu';

interface Props {
  isSelected: boolean;
}

const selector = createSelector(
  stateSelector,
  ({ gallery, system }) => {
    const { autoAddBoardId, autoAssignBoardOnClick } = gallery;
    const { isProcessing } = system;
    return { autoAddBoardId, autoAssignBoardOnClick, isProcessing };
  },
  defaultSelectorOptions
);

const NoBoardBoard = memo(({ isSelected }: Props) => {
  const dispatch = useAppDispatch();
  const { autoAddBoardId, autoAssignBoardOnClick, isProcessing } =
    useAppSelector(selector);
  const boardName = useBoardName('none');
  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected('none'));
    if (autoAssignBoardOnClick && !isProcessing) {
      dispatch(autoAddBoardIdChanged('none'));
    }
  }, [dispatch, autoAssignBoardOnClick, isProcessing]);
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

  return (
    <Box sx={{ w: 'full', h: 'full', touchAction: 'none', userSelect: 'none' }}>
      <Flex
        onMouseOver={handleMouseOver}
        onMouseOut={handleMouseOut}
        sx={{
          position: 'relative',
          justifyContent: 'center',
          alignItems: 'center',
          aspectRatio: '1/1',
          borderRadius: 'base',
          w: 'full',
          h: 'full',
        }}
      >
        <BoardContextMenu board_id="none">
          {(ref) => (
            <Flex
              ref={ref}
              onClick={handleSelectBoard}
              sx={{
                w: 'full',
                h: 'full',
                position: 'relative',
                justifyContent: 'center',
                alignItems: 'center',
                borderRadius: 'base',
                cursor: 'pointer',
                bg: 'base.200',
                _dark: {
                  bg: 'base.800',
                },
              }}
            >
              <Flex
                sx={{
                  w: 'full',
                  h: 'full',
                  justifyContent: 'center',
                  alignItems: 'center',
                }}
              >
                <Image
                  src={InvokeAILogoImage}
                  alt="invoke-ai-logo"
                  sx={{
                    opacity: 0.4,
                    filter: 'grayscale(1)',
                    mt: -6,
                    w: 16,
                    h: 16,
                    minW: 16,
                    minH: 16,
                    userSelect: 'none',
                  }}
                />
              </Flex>
              {autoAddBoardId === 'none' && <AutoAddIcon />}
              <Flex
                sx={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  p: 1,
                  justifyContent: 'center',
                  alignItems: 'center',
                  w: 'full',
                  maxW: 'full',
                  borderBottomRadius: 'base',
                  bg: isSelected ? 'accent.400' : 'base.500',
                  color: isSelected ? 'base.50' : 'base.100',
                  _dark: {
                    bg: isSelected ? 'accent.500' : 'base.600',
                    color: isSelected ? 'base.50' : 'base.100',
                  },
                  lineHeight: 'short',
                  fontSize: 'xs',
                  fontWeight: isSelected ? 700 : 500,
                }}
              >
                {boardName}
              </Flex>
              <SelectionOverlay isSelected={isSelected} isHovered={isHovered} />
              <IAIDroppable
                data={droppableData}
                dropLabel={<Text fontSize="md">Move</Text>}
              />
            </Flex>
          )}
        </BoardContextMenu>
      </Flex>
    </Box>
  );
});

NoBoardBoard.displayName = 'HoverableBoard';

export default memo(NoBoardBoard);
