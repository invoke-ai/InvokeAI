import { Flex } from '@chakra-ui/react';
import WorkflowLibraryButton from 'features/workflowLibrary/components/WorkflowLibraryButton';
import { memo } from 'react';
import WorkflowEditorSettings from './WorkflowEditorSettings';

const TopRightPanel = () => {
  return (
    <Flex sx={{ gap: 2, position: 'absolute', top: 2, insetInlineEnd: 2 }}>
      <WorkflowLibraryButton />
      <WorkflowEditorSettings />
    </Flex>
  );
};

export default memo(TopRightPanel);
