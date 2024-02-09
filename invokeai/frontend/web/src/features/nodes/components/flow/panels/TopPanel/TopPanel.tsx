import { Flex, Spacer } from '@invoke-ai/ui-library';
import AddNodeButton from 'features/nodes/components/flow/panels/TopPanel/AddNodeButton';
import ClearFlowButton from 'features/nodes/components/flow/panels/TopPanel/ClearFlowButton';
import SaveWorkflowButton from 'features/nodes/components/flow/panels/TopPanel/SaveWorkflowButton';
import UpdateNodesButton from 'features/nodes/components/flow/panels/TopPanel/UpdateNodesButton';
import WorkflowName from 'features/nodes/components/flow/panels/TopPanel/WorkflowName';
import WorkflowLibraryMenu from 'features/workflowLibrary/components/WorkflowLibraryMenu/WorkflowLibraryMenu';
import { memo } from 'react';

const TopCenterPanel = () => {
  return (
    <Flex gap={2} top={2} left={2} right={2} position="absolute" alignItems="flex-start" pointerEvents="none">
      <Flex gap="2">
        <AddNodeButton />
        <UpdateNodesButton />
      </Flex>
      <Spacer />
      <WorkflowName />
      <Spacer />
      <ClearFlowButton />
      <SaveWorkflowButton />
      <WorkflowLibraryMenu />
    </Flex>
  );
};

export default memo(TopCenterPanel);
