import { Box, ButtonGroup, Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ClearQueueButton from './ClearQueueButton';
import PauseProcessorButton from './PauseProcessorButton';
import PruneQueueButton from './PruneQueueButton';
import QueueList from './QueueList/QueueList';
import QueueStatus from './QueueStatus';
import ResumeProcessorButton from './ResumeProcessorButton';

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
        <Flex layerStyle="second" borderRadius="base" p={2} gap={2}>
          <ButtonGroup w={28} orientation="vertical" isAttached size="sm">
            <ResumeProcessorButton />
            <PauseProcessorButton />
          </ButtonGroup>
          <ButtonGroup w={28} orientation="vertical" isAttached size="sm">
            <PruneQueueButton />
            <ClearQueueButton />
          </ButtonGroup>
        </Flex>
        <Flex
          layerStyle="second"
          borderRadius="base"
          flexDir="column"
          py={2}
          px={3}
          gap={2}
        >
          <QueueStatus />
        </Flex>
        {/* <QueueStatusCard />
        <CurrentQueueItemCard />
        <NextQueueItemCard /> */}
      </Flex>
      <Box layerStyle="second" p={2} borderRadius="base" w="full" h="full">
        <QueueList />
      </Box>
    </Flex>
  );
};

export default memo(QueueTabContent);
