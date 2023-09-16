import { Box, Flex } from '@chakra-ui/react';
import VerticalQueueControls from 'features/queue/components/VerticalQueueControls';
import { memo } from 'react';
import CurrentQueueItemCard from './CurrentQueueItemCard';
import NextQueueItemCard from './NextQueueItemCard';
import QueueList from './QueueList/QueueList';
import QueueStatusCard from './QueueStatusCard';

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
          <VerticalQueueControls orientation="vertical" />
        </Flex>
        <QueueStatusCard />
        <CurrentQueueItemCard />
        <NextQueueItemCard />
      </Flex>
      <Box layerStyle="second" p={2} borderRadius="base" w="full" h="full">
        <QueueList />
      </Box>
    </Flex>
  );
};

export default memo(QueueTabContent);
