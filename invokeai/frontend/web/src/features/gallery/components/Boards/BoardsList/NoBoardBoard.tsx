import { Badge, Box, ChakraProps, Flex, Icon, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { MoveBoardDropData } from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDroppable from 'common/components/IAIDroppable';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { FaFolder, FaPlus } from 'react-icons/fa';
import { useBoardTotal } from 'services/api/hooks/useBoardTotal';
import AutoAddIcon from '../AutoAddIcon';
import BoardContextMenu from '../BoardContextMenu';

const BASE_BADGE_STYLES: ChakraProps['sx'] = {
  bg: 'base.500',
  color: 'whiteAlpha.900',
};
interface Props {
  isSelected: boolean;
}

const selector = createSelector(
  stateSelector,
  ({ gallery }) => {
    const { autoAddBoardId } = gallery;
    return { autoAddBoardId };
  },
  defaultSelectorOptions
);

const NoBoardBoard = memo(({ isSelected }: Props) => {
  const dispatch = useAppDispatch();
  const { totalImages, totalAssets } = useBoardTotal(undefined);
  const { autoAddBoardId } = useAppSelector(selector);
  const handleSelectBoard = useCallback(() => {
    dispatch(boardIdSelected(undefined));
  }, [dispatch]);

  const droppableData: MoveBoardDropData = useMemo(
    () => ({
      id: 'no_board',
      actionType: 'MOVE_BOARD',
      context: { boardId: undefined },
    }),
    []
  );

  return (
    <Box sx={{ w: 'full', h: 'full', touchAction: 'none', userSelect: 'none' }}>
      <Flex
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
        <BoardContextMenu>
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
                <Icon
                  boxSize={12}
                  as={FaFolder}
                  sx={{
                    opacity: 0.7,
                    color: 'base.500',
                    _dark: {
                      color: 'base.500',
                    },
                  }}
                />
              </Flex>
              <Flex
                sx={{
                  position: 'absolute',
                  insetInlineEnd: 0,
                  top: 0,
                  p: 1,
                }}
              >
                <Badge variant="solid" sx={BASE_BADGE_STYLES}>
                  {totalImages}/{totalAssets}
                </Badge>
              </Flex>
              {!autoAddBoardId && <AutoAddIcon />}
              <Box
                className="selection-box"
                sx={{
                  position: 'absolute',
                  top: 0,
                  insetInlineEnd: 0,
                  bottom: 0,
                  insetInlineStart: 0,
                  borderRadius: 'base',
                  transitionProperty: 'common',
                  transitionDuration: 'common',
                  shadow: isSelected ? 'selected.light' : undefined,
                  _dark: {
                    shadow: isSelected ? 'selected.dark' : undefined,
                  },
                }}
              />
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

export default NoBoardBoard;
