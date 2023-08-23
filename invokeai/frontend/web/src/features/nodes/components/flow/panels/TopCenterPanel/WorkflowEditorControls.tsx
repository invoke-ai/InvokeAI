import { Flex } from '@chakra-ui/react';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import InvokeButton from 'features/parameters/components/ProcessButtons/InvokeButton';
import { memo } from 'react';
import LoadWorkflowButton from './LoadWorkflowButton';
import ResetWorkflowButton from './ResetWorkflowButton';
import SaveWorkflowButton from './SaveWorkflowButton';

const WorkflowEditorControls = () => {
  return (
    <Flex sx={{ gap: 2 }}>
      <InvokeButton />
      <CancelButton />
      <ResetWorkflowButton />
      <SaveWorkflowButton />
      <LoadWorkflowButton />
    </Flex>
  );
};

export default memo(WorkflowEditorControls);
