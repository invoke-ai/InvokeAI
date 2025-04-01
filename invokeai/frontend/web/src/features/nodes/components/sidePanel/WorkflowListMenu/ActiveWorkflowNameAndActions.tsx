import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { WorkflowListMenuTrigger } from 'features/nodes/components/sidePanel/WorkflowListMenu/WorkflowListMenuTrigger';
import { WorkflowViewEditToggleButton } from 'features/nodes/components/sidePanel/WorkflowViewEditToggleButton';
import { selectWorkflowIsPublished, selectWorkflowMode } from 'features/nodes/store/workflowSlice';
import { WorkflowLibraryMenu } from 'features/workflowLibrary/components/WorkflowLibraryMenu/WorkflowLibraryMenu';
import { memo } from 'react';

import SaveWorkflowButton from './SaveWorkflowButton';

export const ActiveWorkflowNameAndActions = memo(() => {
  const mode = useAppSelector(selectWorkflowMode);
  const isPublished = useAppSelector(selectWorkflowIsPublished);

  return (
    <Flex w="full" alignItems="center" gap={1} minW={0}>
      <WorkflowListMenuTrigger />
      <Spacer />
      {mode === 'edit' && !isPublished && <SaveWorkflowButton />}
      <WorkflowViewEditToggleButton />
      <WorkflowLibraryMenu />
    </Flex>
  );
});
ActiveWorkflowNameAndActions.displayName = 'ActiveWorkflowNameAndActions';
