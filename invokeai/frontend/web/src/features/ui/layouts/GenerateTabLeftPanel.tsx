import { Box, Flex } from '@invoke-ai/ui-library';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import QueueControls from 'features/queue/components/QueueControls';
import { ParametersPanelGenerate } from 'features/ui/components/ParametersPanels/ParametersPanelGenerate';
import { memo } from 'react';

export const GenerateTabLeftPanel = memo(() => {
  return (
    <FocusRegionWrapper region="settings" as={Flex} flexDir="column" w="full" h="full" gap={2} p={2}>
      <QueueControls />
      <Box position="relative" w="full" h="full">
        <ParametersPanelGenerate />
      </Box>
    </FocusRegionWrapper>
  );
});
GenerateTabLeftPanel.displayName = 'GenerateTabLeftPanel';
