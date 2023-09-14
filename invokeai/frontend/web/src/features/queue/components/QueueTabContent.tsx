import { Box, Flex } from '@chakra-ui/react';
import QueueControls from 'features/queue/components/QueueControls';
import { memo } from 'react';
import CurrentQueueItemCard from './CurrentQueueItemCard';
import NextQueueItemCard from './NextQueueItemCard';
import QueueStatusCard from './QueueStatusCard';
import QueueTable from './QueueTable';

const QueueTabContent = () => {
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
        <Flex layerStyle="second" borderRadius="base" p={2}>
          <QueueControls orientation="vertical" />
        </Flex>
        <QueueStatusCard />
        <CurrentQueueItemCard />
        <NextQueueItemCard />
      </Flex>
      <Box layerStyle="second" p={2} borderRadius="base" w="full" h="full">
        <QueueTable />
      </Box>
    </Flex>
  );
};

export default memo(QueueTabContent);
