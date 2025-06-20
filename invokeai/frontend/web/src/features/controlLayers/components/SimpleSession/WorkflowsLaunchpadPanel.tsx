import { Flex, Heading } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const WorkflowsLaunchpadPanel = memo(() => {
  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center" justifyContent="center" gap={2}>
      <Flex flexDir="column" w="full" h="full" justifyContent="center" gap={4} px={12} maxW={768}>
        <Heading mb={4}>Go deep with Workflows.</Heading>
      </Flex>
    </Flex>
  );
});
WorkflowsLaunchpadPanel.displayName = 'WorkflowsLaunchpadPanel';
