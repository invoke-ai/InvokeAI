import { Box, Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';

import InvocationCacheStatus from './InvocationCacheStatus';
import { QueueList } from './QueueList/QueueList';
import QueueStatus from './QueueStatus';
import QueueTabQueueControls from './QueueTabQueueControls';

const QueueTabContent = () => {
  return (
    <Flex borderRadius="base" w="full" h="full" flexDir="column" gap={2}>
      <Flex gap={2} w="full">
        <QueueTabQueueControls />
        <QueueStatus />
        <InvocationCacheStatus />
      </Flex>
      <Box layerStyle="first" p={2} borderRadius="base" w="full" h="full">
        <QueueList />
      </Box>
    </Flex>
  );
};

export default memo(QueueTabContent);
