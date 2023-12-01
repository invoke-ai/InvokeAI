import { Flex } from '@chakra-ui/layout';
import { memo } from 'react';
import DownloadWorkflowButton from 'features/workflowLibrary/components/DownloadWorkflowButton';
import UploadWorkflowButton from 'features/workflowLibrary/components/LoadWorkflowFromFileButton';
import ResetWorkflowButton from 'features/workflowLibrary/components/ResetWorkflowButton';
import SaveWorkflowButton from 'features/workflowLibrary/components/SaveWorkflowButton';
import SaveWorkflowAsButton from 'features/workflowLibrary/components/SaveWorkflowAsButton';

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
      <UploadWorkflowButton />
      <SaveWorkflowButton />
      <SaveWorkflowAsButton />
      <ResetWorkflowButton />
    </Flex>
  );
};

export default memo(TopCenterPanel);
