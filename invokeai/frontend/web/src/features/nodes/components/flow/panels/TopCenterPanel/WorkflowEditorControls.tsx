import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import InvokeButton from 'features/parameters/components/ProcessButtons/InvokeButton';
import { memo } from 'react';
import ResetWorkflowButton from './ResetWorkflowButton';
import LoadWorkflowButton from './LoadWorkflowButton';
import SaveWorkflowButton from './SaveWorkflowButton';

const WorkflowEditorControls = () => {
  return (
    <>
      <InvokeButton />
      <CancelButton />
      <ResetWorkflowButton />
      <SaveWorkflowButton />
      <LoadWorkflowButton />
    </>
  );
};

export default memo(WorkflowEditorControls);
