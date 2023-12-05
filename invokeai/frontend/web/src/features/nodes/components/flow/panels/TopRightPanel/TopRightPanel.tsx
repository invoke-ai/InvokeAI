import { Flex } from '@chakra-ui/react';
import WorkflowLibraryButton from 'features/workflowLibrary/components/WorkflowLibraryButton';
import { memo } from 'react';
import WorkflowEditorSettings from './WorkflowEditorSettings';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';

const TopRightPanel = () => {
  const isWorkflowLibraryEnabled =
    useFeatureStatus('workflowLibrary').isFeatureEnabled;

  return (
    <Flex sx={{ gap: 2, position: 'absolute', top: 2, insetInlineEnd: 2 }}>
      {isWorkflowLibraryEnabled && <WorkflowLibraryButton />}
      <WorkflowEditorSettings />
    </Flex>
  );
};

export default memo(TopRightPanel);
