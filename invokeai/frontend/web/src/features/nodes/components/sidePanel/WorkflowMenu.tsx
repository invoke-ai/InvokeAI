import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import SaveWorkflowButton from 'features/nodes/components/flow/panels/TopPanel/SaveWorkflowButton';
import { NewWorkflowButton } from 'features/workflowLibrary/components/NewWorkflowButton';

import { ModeToggle } from './ModeToggle';

export const WorkflowMenu = () => {
  const mode = useAppSelector((s) => s.workflow.mode);

  return (
    <Flex gap="2" alignItems="center">
      {mode === 'edit' && <SaveWorkflowButton />}
      <NewWorkflowButton />
      <ModeToggle />
    </Flex>
  );
};
