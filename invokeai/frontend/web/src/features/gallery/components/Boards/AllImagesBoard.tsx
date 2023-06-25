import { Flex, Text } from '@chakra-ui/react';
import { FaImages } from 'react-icons/fa';
import { boardIdSelected } from '../../store/boardSlice';
import { useDispatch } from 'react-redux';
import { IAINoImageFallback } from 'common/components/IAIImageFallback';
import { AnimatePresence } from 'framer-motion';
import { SelectedItemOverlay } from '../SelectedItemOverlay';
import { useCallback } from 'react';
import { ImageDTO } from 'services/api/types';
import { useRemoveImageFromBoardMutation } from 'services/api/endpoints/boardImages';
import { useDroppable } from '@dnd-kit/core';
import IAIDropOverlay from 'common/components/IAIDropOverlay';

const AllImagesBoard = ({ isSelected }: { isSelected: boolean }) => {
  const dispatch = useDispatch();

  const handleAllImagesBoardClick = () => {
    dispatch(boardIdSelected());
  };

  const [removeImageFromBoard, { isLoading }] =
    useRemoveImageFromBoardMutation();

  const handleDrop = useCallback(
    (droppedImage: ImageDTO) => {
      if (!droppedImage.board_id) {
        return;
      }
      removeImageFromBoard({
        board_id: droppedImage.board_id,
        image_name: droppedImage.image_name,
      });
    },
    [removeImageFromBoard]
  );

  const {
    isOver,
    setNodeRef,
    active: isDropActive,
  } = useDroppable({
    id: `board_droppable_all_images`,
    data: {
      handleDrop,
    },
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
      onClick={handleAllImagesBoardClick}
    >
      <Flex
        ref={setNodeRef}
        sx={{
          position: 'relative',
          justifyContent: 'center',
          alignItems: 'center',
          borderRadius: 'base',
          w: 'full',
          aspectRatio: '1/1',
        }}
      >
        <IAINoImageFallback iconProps={{ boxSize: 8 }} as={FaImages} />
        <AnimatePresence>
          {isSelected && <SelectedItemOverlay />}
        </AnimatePresence>
        <AnimatePresence>
          {isDropActive && <IAIDropOverlay isOver={isOver} />}
        </AnimatePresence>
      </Flex>
      <Text
        sx={{
          color: isSelected ? 'base.50' : 'base.200',
          fontWeight: isSelected ? 600 : undefined,
          fontSize: 'xs',
        }}
      >
        All Images
      </Text>
    </Flex>
  );
};

export default AllImagesBoard;
