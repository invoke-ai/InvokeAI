import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { EditModeLeftPanelContent } from 'features/nodes/components/sidePanel/EditModeLeftPanelContent';
import { PublishedWorkflowPanelContent } from 'features/nodes/components/sidePanel/PublishedWorkflowPanelContent';
import { $isInPublishFlow, useIsWorkflowPublished } from 'features/nodes/components/sidePanel/workflow/publish';
import { PublishWorkflowPanelContent } from 'features/nodes/components/sidePanel/workflow/PublishWorkflowPanelContent';
import { ActiveWorkflowDescription } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowDescription';
import { ActiveWorkflowNameAndActions } from 'features/nodes/components/sidePanel/WorkflowListMenu/ActiveWorkflowNameAndActions';
import { selectWorkflowMode } from 'features/nodes/store/workflowLibrarySlice';
import QueueControls from 'features/queue/components/QueueControls';
import { memo } from 'react';

import { ViewModeLeftPanelContent } from './viewMode/ViewModeLeftPanelContent';

const WorkflowsTabLeftPanel = () => {
  const mode = useAppSelector(selectWorkflowMode);
  const isPublished = useIsWorkflowPublished();
  const isInPublishFlow = useStore($isInPublishFlow);

  return (
    <FocusRegionWrapper region="settings" as={Flex} flexDir="column" w="full" h="full" gap={2} p={2}>
      <QueueControls />
      <Flex w="full" h="full" gap={2} flexDir="column">
        {isInPublishFlow && <PublishWorkflowPanelContent />}
        {!isInPublishFlow && <ActiveWorkflowNameAndActions />}
        {!isInPublishFlow && !isPublished && mode === 'view' && <ActiveWorkflowDescription />}
        {!isInPublishFlow && !isPublished && mode === 'view' && <ViewModeLeftPanelContent />}
        {!isInPublishFlow && !isPublished && mode === 'edit' && <EditModeLeftPanelContent />}
        {isPublished && <PublishedWorkflowPanelContent />}
      </Flex>
    </FocusRegionWrapper>
  );
};

export default memo(WorkflowsTabLeftPanel);
