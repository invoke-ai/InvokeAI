import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { EditModeLeftPanelContent } from 'features/nodes/components/sidePanel/EditModeLeftPanelContent';
import { $isInPublishFlow } from 'features/nodes/components/sidePanel/workflow/publish';
import { PublishWorkflowPanelContent } from 'features/nodes/components/sidePanel/workflow/PublishWorkflowPanelContent';
import { ActiveWorkflowDescription } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowDescription';
import { ActiveWorkflowNameAndActions } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowNameAndActions';
import { selectWorkflowMode } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';

import { ViewModeLeftPanelContent } from './viewMode/ViewModeLeftPanelContent';

const WorkflowsTabLeftPanel = () => {
  const mode = useAppSelector(selectWorkflowMode);
  const isInPublishFlow = useStore($isInPublishFlow);

  return (
    <Flex w="full" h="full" gap={2} flexDir="column">
      {isInPublishFlow && <PublishWorkflowPanelContent />}
      {!isInPublishFlow && <ActiveWorkflowNameAndActions />}
      {!isInPublishFlow && mode === 'view' && <ActiveWorkflowDescription />}
      {!isInPublishFlow && mode === 'view' && <ViewModeLeftPanelContent />}
      {!isInPublishFlow && mode === 'edit' && <EditModeLeftPanelContent />}
    </Flex>
  );
};

export default memo(WorkflowsTabLeftPanel);
