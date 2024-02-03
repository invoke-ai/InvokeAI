import { Flex } from '@invoke-ai/ui-library';
import WorkflowLibraryButton from 'features/workflowLibrary/components/WorkflowLibraryButton';
import WorkflowLibraryMenu from 'features/workflowLibrary/components/WorkflowLibraryMenu/WorkflowLibraryMenu';
import { memo } from 'react';

const TopRightPanel = () => {
  return (
    <Flex gap={2} position="absolute" top={2} insetInlineEnd={2}>
      <WorkflowLibraryButton />
      <WorkflowLibraryMenu />
    </Flex>
  );
};

export default memo(TopRightPanel);
