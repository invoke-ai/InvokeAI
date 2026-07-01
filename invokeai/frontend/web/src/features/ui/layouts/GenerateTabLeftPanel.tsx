import { Box, Flex } from '@invoke-ai/ui-library';
import QueueControls from 'features/queue/components/QueueControls';
import { ParametersPanelGenerate } from 'features/ui/components/ParametersPanels/ParametersPanelGenerate';
import { memo } from 'react';

export const GenerateTabLeftPanel = memo(() => {
  return (
    <Flex flexDir="column" w="full" h="full" gap={2}>
      <QueueControls />
      <Box position="relative" w="full" h="full">
        <ParametersPanelGenerate />
      </Box>
    </Flex>
  );
});
GenerateTabLeftPanel.displayName = 'GenerateTabLeftPanel';
