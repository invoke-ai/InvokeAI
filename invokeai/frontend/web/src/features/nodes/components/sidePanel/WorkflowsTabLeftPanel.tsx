import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { $isInDeployFlow } from 'features/nodes/components/sidePanel/builder/deploy';
import { EditModeLeftPanelContent } from 'features/nodes/components/sidePanel/EditModeLeftPanelContent';
import { DeployWorkflowPanelContent } from 'features/nodes/components/sidePanel/workflow/DeployWorkflowPanelContent';
import { ActiveWorkflowDescription } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowDescription';
import { ActiveWorkflowNameAndActions } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowNameAndActions';
import { selectWorkflowMode } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';

import { ViewModeLeftPanelContent } from './viewMode/ViewModeLeftPanelContent';

const WorkflowsTabLeftPanel = () => {
  const mode = useAppSelector(selectWorkflowMode);
  const isInDeployFlow = useStore($isInDeployFlow);

  return (
    <Flex w="full" h="full" gap={2} flexDir="column">
      {isInDeployFlow && <DeployWorkflowPanelContent />}
      {!isInDeployFlow && <ActiveWorkflowNameAndActions />}
      {!isInDeployFlow && mode === 'view' && <ActiveWorkflowDescription />}
      {!isInDeployFlow && mode === 'view' && <ViewModeLeftPanelContent />}
      {!isInDeployFlow && mode === 'edit' && <EditModeLeftPanelContent />}
    </Flex>
  );
};

export default memo(WorkflowsTabLeftPanel);
