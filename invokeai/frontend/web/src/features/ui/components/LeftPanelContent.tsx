import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import WorkflowsTabLeftPanel from 'features/nodes/components/sidePanel/WorkflowsTabLeftPanel';
import QueueControls from 'features/queue/components/QueueControls';
import ParametersPanelTextToImage from 'features/ui/components/ParametersPanels/ParametersPanelTextToImage';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

import ParametersPanelUpscale from './ParametersPanels/ParametersPanelUpscale';

const LeftPanelContent = memo(() => {
  const tab = useAppSelector(selectActiveTab);

  return (
    <Flex flexDir="column" w="full" h="full" gap={2}>
      <QueueControls />
      <Box position="relative" w="full" h="full">
        {tab === 'canvas' && <ParametersPanelTextToImage />}
        {tab === 'upscaling' && <ParametersPanelUpscale />}
        {tab === 'workflows' && <WorkflowsTabLeftPanel />}
      </Box>
    </Flex>
  );
});
LeftPanelContent.displayName = 'LeftPanelContent';
