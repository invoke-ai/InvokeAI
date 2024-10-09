import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import AddNodeButton from 'features/nodes/components/flow/panels/TopPanel/AddNodeButton';
import ClearFlowButton from 'features/nodes/components/flow/panels/TopPanel/ClearFlowButton';
import SaveWorkflowButton from 'features/nodes/components/flow/panels/TopPanel/SaveWorkflowButton';
import UpdateNodesButton from 'features/nodes/components/flow/panels/TopPanel/UpdateNodesButton';
import { WorkflowName } from 'features/nodes/components/sidePanel/WorkflowName';
import { selectWorkflowName } from 'features/nodes/store/workflowSlice';
import WorkflowLibraryMenu from 'features/workflowLibrary/components/WorkflowLibraryMenu/WorkflowLibraryMenu';
import { memo } from 'react';

const TopCenterPanel = () => {
  const name = useAppSelector(selectWorkflowName);
  return (
    <Flex gap={2} top={0} left={0} right={0} position="absolute" alignItems="flex-start" pointerEvents="none">
      <Flex gap="2">
        <AddNodeButton />
        <UpdateNodesButton />
      </Flex>
      <Spacer />
      {!!name.length && <WorkflowName />}
      <Spacer />
      <ClearFlowButton />
      <SaveWorkflowButton />
      <WorkflowLibraryMenu />
    </Flex>
  );
};

export default memo(TopCenterPanel);
