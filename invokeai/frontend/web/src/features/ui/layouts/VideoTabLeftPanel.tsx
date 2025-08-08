import { Box, Flex } from '@invoke-ai/ui-library';
import QueueControls from 'features/queue/components/QueueControls';
import { memo } from 'react';
import { ParametersPanelVideo } from '../components/ParametersPanels/ParametersPanelVideo';

export const VideoTabLeftPanel = memo(() => {
  return (
    <Flex flexDir="column" w="full" h="full" gap={2}>
      <QueueControls />
      <Box position="relative" w="full" h="full">
        <ParametersPanelVideo />
      </Box>
    </Flex>
  );
});
VideoTabLeftPanel.displayName = 'VideoTabLeftPanel';
