import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import WorkflowsTabLeftPanel from 'features/nodes/components/sidePanel/WorkflowsTabLeftPanel';
import ParametersPanelTextToImage from 'features/ui/components/ParametersPanels/ParametersPanelTextToImage';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

import ParametersPanelUpscale from './ParametersPanels/ParametersPanelUpscale';

export const LeftPanelContent = memo(() => {
  const tab = useAppSelector(selectActiveTab);

  return (
    <Box position="relative" w="full" h="full" p={2}>
      {tab === 'canvas' && <ParametersPanelTextToImage />}
      {tab === 'upscaling' && <ParametersPanelUpscale />}
      {tab === 'workflows' && <WorkflowsTabLeftPanel />}
    </Box>
  );
});
LeftPanelContent.displayName = 'LeftPanelContent';
