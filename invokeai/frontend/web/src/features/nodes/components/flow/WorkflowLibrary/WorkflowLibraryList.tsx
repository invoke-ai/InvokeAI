import { Flex } from '@chakra-ui/react';
import WorkflowLibraryWorkflowItem from 'features/nodes/components/flow/WorkflowLibrary/WorkflowLibraryWorkflowItem';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import { memo } from 'react';
import { paths } from 'services/api/schema';

type Props = {
  data: paths['/api/v1/workflows/']['get']['responses']['200']['content']['application/json'];
};

const WorkflowLibraryList = ({ data }: Props) => {
  return (
    <Flex w="full" h="full" layerStyle="second" p={2} borderRadius="base">
      <ScrollableContent>
        <Flex w="full" h="full" gap={2} flexDir="column">
          {data.items.map((w) => (
            <WorkflowLibraryWorkflowItem key={w.workflow_id} workflowDTO={w} />
          ))}
        </Flex>
      </ScrollableContent>
    </Flex>
  );
};

export default memo(WorkflowLibraryList);
