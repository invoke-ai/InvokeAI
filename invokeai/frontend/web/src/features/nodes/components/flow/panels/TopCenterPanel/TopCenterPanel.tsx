import { Flex, Heading } from '@chakra-ui/layout';
import { memo } from 'react';
import DownloadWorkflowButton from 'features/workflowLibrary/components/DownloadWorkflowButton';
import UploadWorkflowButton from 'features/workflowLibrary/components/LoadWorkflowFromFileButton';
import ResetWorkflowEditorButton from 'features/workflowLibrary/components/ResetWorkflowButton';
import SaveWorkflowButton from 'features/workflowLibrary/components/SaveWorkflowButton';
import SaveWorkflowAsButton from 'features/workflowLibrary/components/SaveWorkflowAsButton';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useAppSelector } from 'app/store/storeHooks';

const TopCenterPanel = () => {
  const isWorkflowLibraryEnabled =
    useFeatureStatus('workflowLibrary').isFeatureEnabled;
  const name = useAppSelector(
    (state) => state.workflow.name || 'Untitled Workflow'
  );

  return (
    <Flex
      sx={{
        gap: 2,
        position: 'absolute',
        top: 2,
        insetInlineStart: '50%',
        transform: 'translate(-50%)',
        alignItems: 'center',
      }}
    >
      <Heading
        m={2}
        size="md"
        userSelect="none"
        pointerEvents="none"
        noOfLines={1}
        wordBreak="break-all"
        maxW={80}
      >
        {name}
      </Heading>
      {/* <DownloadWorkflowButton />
      <UploadWorkflowButton />
      {isWorkflowLibraryEnabled && (
        <>
          <SaveWorkflowButton />
          <SaveWorkflowAsButton />
        </>
      )}
      <ResetWorkflowEditorButton /> */}
    </Flex>
  );
};

export default memo(TopCenterPanel);
