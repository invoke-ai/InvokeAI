import { Flex, useColorMode } from '@chakra-ui/react';
import { FaImages } from 'react-icons/fa';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { useDispatch } from 'react-redux';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { AnimatePresence } from 'framer-motion';
import IAIDropOverlay from 'common/components/IAIDropOverlay';
import { mode } from 'theme/util/mode';
import {
  MoveBoardDropData,
  isValidDrop,
  useDroppable,
} from 'app/components/ImageDnd/typesafeDnd';

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

  const { isOver, setNodeRef, active } = useDroppable({
    id: `board_droppable_all_images`,
    data: droppableData,
  });

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
        ref={setNodeRef}
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
        <AnimatePresence>
          {isValidDrop(droppableData, active) && (
            <IAIDropOverlay isOver={isOver} />
          )}
        </AnimatePresence>
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
