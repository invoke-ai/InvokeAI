import { Box, Flex } from '@invoke-ai/ui-library';
import QueueControls from 'features/queue/components/QueueControls';
import { ParametersPanelCanvas } from 'features/ui/components/ParametersPanels/ParametersPanelCanvas';
import { memo } from 'react';

export const CanvasTabLeftPanel = memo(() => {
  return (
    <Flex flexDir="column" w="full" h="full" gap={2} py={2} pe={2}>
      <QueueControls />
      <Box position="relative" w="full" h="full">
        <ParametersPanelCanvas />
      </Box>
    </Flex>
  );
});
CanvasTabLeftPanel.displayName = 'CanvasTabLeftPanel';
