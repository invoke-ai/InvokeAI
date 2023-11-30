import { Flex } from '@chakra-ui/layout';
import { memo } from 'react';
import DownloadWorkflowButton from './DownloadWorkflowButton';
import LoadWorkflowButton from './LoadWorkflowButton';
import ResetWorkflowButton from './ResetWorkflowButton';
import SaveWorkflowButton from 'features/nodes/components/flow/panels/TopCenterPanel/SaveWorkflowButton';

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
      <SaveWorkflowButton />
      <ResetWorkflowButton />
    </Flex>
  );
};

export default memo(TopCenterPanel);
