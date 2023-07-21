import { Badge, Box, ChakraProps, Flex, Icon, Text } from '@chakra-ui/react';
import { MoveBoardDropData } from 'app/components/ImageDnd/typesafeDnd';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { FaFolder } from 'react-icons/fa';
import { useBoardTotal } from 'services/api/hooks/useBoardTotal';

const BASE_BADGE_STYLES: ChakraProps['sx'] = {
  bg: 'base.500',
  color: 'whiteAlpha.900',
};
interface Props {
  isSelected: boolean;
}

const NoBoardBoard = memo(({ isSelected }: Props) => {
  const dispatch = useAppDispatch();
  const { totalImages, totalAssets } = useBoardTotal(undefined);
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
        onClick={handleSelectBoard}
        sx={{
          position: 'relative',
          justifyContent: 'center',
          alignItems: 'center',
          aspectRatio: '1/1',
          borderRadius: 'base',
          cursor: 'pointer',
          w: 'full',
          h: 'full',
        }}
      >
        <Flex
          sx={{
            w: 'full',
            h: 'full',
            justifyContent: 'center',
            alignItems: 'center',
            borderRadius: 'base',
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
    </Box>
  );
});

NoBoardBoard.displayName = 'HoverableBoard';

export default NoBoardBoard;
