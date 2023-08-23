import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import InvokeButton from 'features/parameters/components/ProcessButtons/InvokeButton';
import { memo } from 'react';
import ClearGraphButton from './ClearGraphButton';
import LoadWorkflowButton from './LoadWorkflowButton';

const WorkflowEditorControls = () => {
  return (
    <>
      <InvokeButton />
      <CancelButton />
      <ClearGraphButton />
      <LoadWorkflowButton />
    </>
  );
};

export default memo(WorkflowEditorControls);
