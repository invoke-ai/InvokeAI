import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

import InvocationCacheStatus from './InvocationCacheStatus';
import QueueList from './QueueList/QueueList';
import QueueStatus from './QueueStatus';
import QueueTabQueueControls from './QueueTabQueueControls';

const QueueTabContent = () => {
  const isInvocationCacheEnabled = useFeatureStatus('invocationCache');
  const activeTabName = useAppSelector(selectActiveTab);

  return (
    <Flex
      display={activeTabName === 'queue' ? undefined : 'none'}
      hidden={activeTabName !== 'queue'}
      borderRadius="base"
      w="full"
      h="full"
      flexDir="column"
      gap={2}
    >
      <Flex gap={2} w="full">
        <QueueTabQueueControls />
        <QueueStatus />
        {isInvocationCacheEnabled && <InvocationCacheStatus />}
      </Flex>
      <Box layerStyle="first" p={2} borderRadius="base" w="full" h="full">
        <QueueList />
      </Box>
    </Flex>
  );
};

export default memo(QueueTabContent);
