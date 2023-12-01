import WorkflowLibraryList from 'features/workflowLibrary/components/WorkflowLibraryList';
import WorkflowLibraryListWrapper from 'features/workflowLibrary/components/WorkflowLibraryListWrapper';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const WorkflowLibraryContent = () => {
  const { t } = useTranslation();

  return (
    <WorkflowLibraryListWrapper>
      <WorkflowLibraryList />
    </WorkflowLibraryListWrapper>
  );
};

export default memo(WorkflowLibraryContent);
