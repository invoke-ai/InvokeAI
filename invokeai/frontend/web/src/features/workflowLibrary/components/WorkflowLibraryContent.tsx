import WorkflowLibraryList from 'features/workflowLibrary/components/WorkflowLibraryList';
import WorkflowLibraryListWrapper from 'features/workflowLibrary/components/WorkflowLibraryListWrapper';
import { memo } from 'react';

const WorkflowLibraryContent = () => {
  return (
    <WorkflowLibraryListWrapper>
      <WorkflowLibraryList />
    </WorkflowLibraryListWrapper>
  );
};

export default memo(WorkflowLibraryContent);
