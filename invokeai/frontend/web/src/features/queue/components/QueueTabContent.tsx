import { Box, Flex } from '@chakra-ui/react';
import { memo } from 'react';
import { useFeatureStatus } from '../../system/hooks/useFeatureStatus';
import InvocationCacheStatus from './InvocationCacheStatus';
import QueueList from './QueueList/QueueList';
import QueueStatus from './QueueStatus';
import QueueTabQueueControls from './QueueTabQueueControls';

const QueueTabContent = () => {
  const isInvocationCacheEnabled =
    useFeatureStatus('invocationCache').isFeatureEnabled;

  return (
    <Flex
      layerStyle="first"
      borderRadius="base"
      w="full"
      h="full"
      p={2}
      flexDir="column"
      gap={2}
    >
      <Flex gap={2} w="full">
        <QueueTabQueueControls />
        <QueueStatus />
        {isInvocationCacheEnabled && <InvocationCacheStatus />}
      </Flex>
      <Box layerStyle="second" p={2} borderRadius="base" w="full" h="full">
        <QueueList />
      </Box>
    </Flex>
  );
};

export default memo(QueueTabContent);
