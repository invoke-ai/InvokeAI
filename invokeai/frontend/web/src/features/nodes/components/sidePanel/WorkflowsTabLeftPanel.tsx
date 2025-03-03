import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { EditModeLeftPanelContent } from 'features/nodes/components/sidePanel/EditModeLeftPanelContent';
import { ActiveWorkflowDescription } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowDescription';
import { ActiveWorkflowNameAndActions } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowNameAndActions';
import { selectWorkflowMode } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';

import { ViewModeLeftPanelContent } from './viewMode/ViewModeLeftPanelContent';

const WorkflowsTabLeftPanel = () => {
  const mode = useAppSelector(selectWorkflowMode);

  return (
    <Flex w="full" h="full" gap={2} flexDir="column">
      <ActiveWorkflowNameAndActions />
      {mode === 'view' && <ActiveWorkflowDescription />}
      {mode === 'view' && <ViewModeLeftPanelContent />}
      {mode === 'edit' && <EditModeLeftPanelContent />}
    </Flex>
  );
};

export default memo(WorkflowsTabLeftPanel);
