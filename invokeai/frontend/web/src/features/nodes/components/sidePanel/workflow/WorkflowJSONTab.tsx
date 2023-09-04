import { Flex } from '@chakra-ui/react';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { useWorkflow } from 'features/nodes/hooks/useWorkflow';
import { memo } from 'react';

const WorkflowJSONTab = () => {
  const workflow = useWorkflow();

  return (
    <Flex
      sx={{
        flexDir: 'column',
        alignItems: 'flex-start',
        gap: 2,
        h: 'full',
      }}
    >
      <DataViewer data={workflow} label="Workflow" />
    </Flex>
  );
};

export default memo(WorkflowJSONTab);
