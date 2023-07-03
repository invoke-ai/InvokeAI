import { Box } from '@chakra-ui/react';
import BatchImageGrid from './BatchImageGrid';
import IAIDropOverlay from 'common/components/IAIDropOverlay';
import {
  AddToBatchDropData,
  isValidDrop,
  useDroppable,
} from 'app/components/ImageDnd/typesafeDnd';

const droppableData: AddToBatchDropData = {
  id: 'batch',
  actionType: 'ADD_TO_BATCH',
};

const BatchImageContainer = () => {
  const { isOver, setNodeRef, active } = useDroppable({
    id: 'batch-manager',
    data: droppableData,
  });

  return (
    <Box ref={setNodeRef} position="relative" w="full" h="full">
      <BatchImageGrid />
      {isValidDrop(droppableData, active) && (
        <IAIDropOverlay isOver={isOver} label="Add to Batch" />
      )}
    </Box>
  );
};

export default BatchImageContainer;
