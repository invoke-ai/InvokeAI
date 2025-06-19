import { Box, Flex } from '@invoke-ai/ui-library';
import QueueControls from 'features/queue/components/QueueControls';
import { ParametersPanelUpscale } from 'features/ui/components/ParametersPanels/ParametersPanelUpscale';
import { memo } from 'react';

export const UpscalingTabLeftPanel = memo(() => {
  return (
    <Flex flexDir="column" w="full" h="full" gap={2} py={2} pe={2}>
      <QueueControls />
      <Box position="relative" w="full" h="full">
        <ParametersPanelUpscale />
      </Box>
    </Flex>
  );
});
UpscalingTabLeftPanel.displayName = 'UpscalingTabLeftPanel';
