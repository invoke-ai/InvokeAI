import { Box, Flex } from '@invoke-ai/ui-library';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import QueueControls from 'features/queue/components/QueueControls';
import { ParametersPanelCanvas } from 'features/ui/components/ParametersPanels/ParametersPanelCanvas';
import { memo } from 'react';

export const CanvasTabLeftPanel = memo(() => {
  return (
    <FocusRegionWrapper region="settings" as={Flex} flexDir="column" w="full" h="full" gap={2} p={2}>
      <QueueControls />
      <Box position="relative" w="full" h="full">
        <ParametersPanelCanvas />
      </Box>
    </FocusRegionWrapper>
  );
});
CanvasTabLeftPanel.displayName = 'CanvasTabLeftPanel';
