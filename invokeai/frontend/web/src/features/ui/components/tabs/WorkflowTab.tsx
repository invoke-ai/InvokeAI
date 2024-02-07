import { Box, Flex } from '@invoke-ai/ui-library';
import CurrentImageDisplay from 'features/gallery/components/CurrentImage/CurrentImageDisplay';
import { memo } from 'react';

const WorkflowTab = () => {
  return (
    <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
      <Flex w="full" h="full">
        <CurrentImageDisplay />
      </Flex>
    </Box>
  );
};

export default memo(WorkflowTab);
