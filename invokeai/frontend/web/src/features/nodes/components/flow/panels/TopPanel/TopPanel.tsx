import { Flex, Spacer } from '@chakra-ui/layout';
import AddNodeButton from 'features/nodes/components/flow/panels/TopPanel/AddNodeButton';
import UpdateNodesButton from 'features/nodes/components/flow/panels/TopPanel/UpdateNodesButton';
import WorkflowName from 'features/nodes/components/flow/panels/TopPanel/WorkflowName';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import WorkflowLibraryButton from 'features/workflowLibrary/components/WorkflowLibraryButton';
import WorkflowLibraryMenu from 'features/workflowLibrary/components/WorkflowLibraryMenu/WorkflowLibraryMenu';
import { memo } from 'react';

const TopCenterPanel = () => {
  const isWorkflowLibraryEnabled =
    useFeatureStatus('workflowLibrary').isFeatureEnabled;

  return (
    <Flex
      gap={2}
      top={2}
      left={2}
      right={2}
      position="absolute"
      alignItems="center"
      pointerEvents="none"
    >
      <AddNodeButton />
      <UpdateNodesButton />
      <Spacer />
      <WorkflowName />
      <Spacer />
      {isWorkflowLibraryEnabled && <WorkflowLibraryButton />}
      <WorkflowLibraryMenu />
    </Flex>
  );
};

export default memo(TopCenterPanel);
