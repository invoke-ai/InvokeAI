import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import NewWorkflowButton from 'features/workflowLibrary/components/NewWorkflowButton';

import { ModeToggle } from './ModeToggle';
import SaveWorkflowButton from '../flow/panels/TopPanel/SaveWorkflowButton';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => {
  return {
    mode: workflow.mode,
  };
});

export const WorkflowMenu = () => {
  const { mode } = useAppSelector(selector);

  return (
    <Flex gap="2" alignItems="center">
      {mode === 'edit' && <SaveWorkflowButton />}
      <NewWorkflowButton />
      <ModeToggle />
    </Flex>
  );
};
