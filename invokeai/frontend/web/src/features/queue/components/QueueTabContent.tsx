import { Box, Flex } from '@invoke-ai/ui-library';
import { useIsAdmin } from 'features/auth/hooks/useIsAdmin';
import { memo } from 'react';

import InvocationCacheStatus from './InvocationCacheStatus';
import { QueueList } from './QueueList/QueueList';
import QueueStatus from './QueueStatus';
import QueueTabQueueControls from './QueueTabQueueControls';

const QueueTabContent = () => {
  // The invocation cache status and its enable/disable/clear mutations are administrator-only on the
  // backend. Rendering the panel for a non-admin would show zeroed stats and offer buttons that 403.
  const isAdmin = useIsAdmin();

  return (
    <Flex borderRadius="base" w="full" h="full" flexDir="column" gap={2}>
      <Flex gap={2} w="full">
        <QueueTabQueueControls />
        <QueueStatus />
        {isAdmin && <InvocationCacheStatus />}
      </Flex>
      <Box layerStyle="first" p={2} borderRadius="base" w="full" h="full">
        <QueueList />
      </Box>
    </Flex>
  );
};

export default memo(QueueTabContent);
