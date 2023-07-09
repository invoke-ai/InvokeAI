import { Flex, useColorMode } from '@chakra-ui/react';
import { MoveBoardDropData } from 'app/components/ImageDnd/typesafeDnd';
import IAIDroppable from 'common/components/IAIDroppable';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { FaImages } from 'react-icons/fa';
import { useDispatch } from 'react-redux';
import { mode } from 'theme/util/mode';

const AllImagesBoard = ({ isSelected }: { isSelected: boolean }) => {
  const dispatch = useDispatch();
  const { colorMode } = useColorMode();

  const handleAllImagesBoardClick = () => {
    dispatch(boardIdSelected());
  };

  const droppableData: MoveBoardDropData = {
    id: 'all-images-board',
    actionType: 'MOVE_BOARD',
    context: { boardId: null },
  };

  return (
    <Flex
      sx={{
        flexDir: 'column',
        justifyContent: 'space-between',
        alignItems: 'center',
        cursor: 'pointer',
        w: 'full',
        h: 'full',
        borderRadius: 'base',
      }}
    >
      <Flex
        onClick={handleAllImagesBoardClick}
        sx={{
          position: 'relative',
          justifyContent: 'center',
          alignItems: 'center',
          borderRadius: 'base',
          w: 'full',
          aspectRatio: '1/1',
          overflow: 'hidden',
          shadow: isSelected ? 'selected.light' : undefined,
          _dark: { shadow: isSelected ? 'selected.dark' : undefined },
          flexShrink: 0,
        }}
      >
        <IAINoContentFallback
          boxSize={8}
          icon={FaImages}
          sx={{
            border: '2px solid var(--invokeai-colors-base-200)',
            _dark: { border: '2px solid var(--invokeai-colors-base-800)' },
          }}
        />
        <IAIDroppable data={droppableData} />
      </Flex>
      <Flex
        sx={{
          h: 'full',
          alignItems: 'center',
          color: isSelected
            ? mode('base.900', 'base.50')(colorMode)
            : mode('base.700', 'base.200')(colorMode),
          fontWeight: isSelected ? 600 : undefined,
          fontSize: 'xs',
        }}
      >
        All Images
      </Flex>
    </Flex>
  );
};

export default AllImagesBoard;
