import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { EditModeLeftPanelContent } from 'features/nodes/components/sidePanel/EditModeLeftPanelContent';
import { PublishedWorkflowPanelContent } from 'features/nodes/components/sidePanel/PublishedWorkflowPanelContent';
import { $isInPublishFlow } from 'features/nodes/components/sidePanel/workflow/publish';
import { PublishWorkflowPanelContent } from 'features/nodes/components/sidePanel/workflow/PublishWorkflowPanelContent';
import { ActiveWorkflowDescription } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowDescription';
import { ActiveWorkflowNameAndActions } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowNameAndActions';
import { selectWorkflowIsPublished, selectWorkflowMode } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';

import { ViewModeLeftPanelContent } from './viewMode/ViewModeLeftPanelContent';

const WorkflowsTabLeftPanel = () => {
  const mode = useAppSelector(selectWorkflowMode);
  const isPublished = useAppSelector(selectWorkflowIsPublished);
  const isInPublishFlow = useStore($isInPublishFlow);

  return (
    <Flex w="full" h="full" gap={2} flexDir="column">
      {isInPublishFlow && <PublishWorkflowPanelContent />}
      {!isInPublishFlow && <ActiveWorkflowNameAndActions />}
      {!isInPublishFlow && !isPublished && mode === 'view' && <ActiveWorkflowDescription />}
      {!isInPublishFlow && !isPublished && mode === 'view' && <ViewModeLeftPanelContent />}
      {!isInPublishFlow && !isPublished && mode === 'edit' && <EditModeLeftPanelContent />}
      {isPublished && <PublishedWorkflowPanelContent />}
    </Flex>
  );
};

export default memo(WorkflowsTabLeftPanel);
