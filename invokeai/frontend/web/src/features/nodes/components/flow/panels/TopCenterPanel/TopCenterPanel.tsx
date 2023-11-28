import { Flex } from '@chakra-ui/layout';
import { memo } from 'react';
import LoadWorkflowButton from './LoadWorkflowButton';
import ResetWorkflowButton from './ResetWorkflowButton';
import DownloadWorkflowButton from './DownloadWorkflowButton';
import WorkflowLibraryButton from 'features/nodes/components/flow/WorkflowLibrary/WorkflowLibraryButton';

const TopCenterPanel = () => {
  return (
    <Flex
      sx={{
        gap: 2,
        position: 'absolute',
        top: 2,
        insetInlineStart: '50%',
        transform: 'translate(-50%)',
      }}
    >
      <DownloadWorkflowButton />
      <LoadWorkflowButton />
      <ResetWorkflowButton />
      <WorkflowLibraryButton />
    </Flex>
  );
};

export default memo(TopCenterPanel);
