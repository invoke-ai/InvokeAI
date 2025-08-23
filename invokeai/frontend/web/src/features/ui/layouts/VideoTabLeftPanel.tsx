import { Box, Flex } from '@invoke-ai/ui-library';
import QueueControls from 'features/queue/components/QueueControls';
import { ParametersPanelVideo } from 'features/ui/components/ParametersPanels/ParametersPanelVideo';
import { memo } from 'react';

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
