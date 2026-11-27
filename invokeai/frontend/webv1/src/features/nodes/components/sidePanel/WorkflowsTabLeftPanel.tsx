import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { EditModeLeftPanelContent } from 'features/nodes/components/sidePanel/EditModeLeftPanelContent';
import { ActiveWorkflowDescription } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowDescription';
import { ActiveWorkflowNameAndActions } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowNameAndActions';
import { selectWorkflowMode } from 'features/nodes/store/workflowLibrarySlice';
import QueueControls from 'features/queue/components/QueueControls';
import { memo } from 'react';

import { ViewModeLeftPanelContent } from './viewMode/ViewModeLeftPanelContent';

const WorkflowsTabLeftPanel = () => {
  const mode = useAppSelector(selectWorkflowMode);

  return (
    <Flex flexDir="column" w="full" h="full" gap={2}>
      <QueueControls />
      <Flex w="full" h="full" gap={2} flexDir="column">
        <ActiveWorkflowNameAndActions />
        {mode === 'view' && <ActiveWorkflowDescription />}
        {mode === 'view' && <ViewModeLeftPanelContent />}
        {mode === 'edit' && <EditModeLeftPanelContent />}
      </Flex>
    </Flex>
  );
};

export default memo(WorkflowsTabLeftPanel);
