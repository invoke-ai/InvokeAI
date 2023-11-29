import { Flex } from '@chakra-ui/layout';
import { memo } from 'react';
import WorkflowEditorSettings from './WorkflowEditorSettings';
import WorkflowLibraryButton from 'features/nodes/components/flow/WorkflowLibrary/WorkflowLibraryButton';

const TopRightPanel = () => {
  return (
    <Flex sx={{ gap: 2, position: 'absolute', top: 2, insetInlineEnd: 2 }}>
      <WorkflowEditorSettings />
      <WorkflowLibraryButton />
    </Flex>
  );
};

export default memo(TopRightPanel);
