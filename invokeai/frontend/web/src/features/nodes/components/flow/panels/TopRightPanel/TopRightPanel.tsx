import { Flex } from '@chakra-ui/react';
import WorkflowLibraryButton from 'features/workflowLibrary/components/WorkflowLibraryButton';
import { memo } from 'react';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import WorkflowLibraryMenu from 'features/workflowLibrary/components/WorkflowLibraryMenu/WorkflowLibraryMenu';

const TopRightPanel = () => {
  const isWorkflowLibraryEnabled =
    useFeatureStatus('workflowLibrary').isFeatureEnabled;

  return (
    <Flex sx={{ gap: 2, position: 'absolute', top: 2, insetInlineEnd: 2 }}>
      {isWorkflowLibraryEnabled && <WorkflowLibraryButton />}
      <WorkflowLibraryMenu />
    </Flex>
  );
};

export default memo(TopRightPanel);
