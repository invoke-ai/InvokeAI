import { Box } from '@chakra-ui/react';
import { AddToBatchDropData } from 'app/components/ImageDnd/typesafeDnd';
import IAIDroppable from 'common/components/IAIDroppable';
import BatchImageGrid from './BatchImageGrid';

const droppableData: AddToBatchDropData = {
  id: 'batch',
  actionType: 'ADD_TO_BATCH',
};

const BatchImageContainer = () => {
  return (
    <Box position="relative" w="full" h="full">
      <BatchImageGrid />
      <IAIDroppable data={droppableData} dropLabel="Add to Batch" />
    </Box>
  );
};

export default BatchImageContainer;
