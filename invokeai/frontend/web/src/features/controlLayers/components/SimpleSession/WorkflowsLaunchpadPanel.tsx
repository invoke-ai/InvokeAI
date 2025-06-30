import { Button, Flex, Heading, Link, Text } from '@invoke-ai/ui-library';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { useNewWorkflow } from 'features/workflowLibrary/components/NewWorkflowConfirmationAlertDialog';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFilePlusBold, PiFolderOpenBold, PiUploadBold } from 'react-icons/pi';

import { LaunchpadButton } from './LaunchpadButton';

export const WorkflowsLaunchpadPanel = memo(() => {
  const { t } = useTranslation();
  const workflowLibraryModal = useWorkflowLibraryModal();
  const newWorkflow = useNewWorkflow();
  const loadWorkflowFromFile = useLoadWorkflowFromFile();

  const onUploadWorkflow = useCallback(
    (file: File) => {
      loadWorkflowFromFile(file);
    },
    [loadWorkflowFromFile]
  );

  const uploadApi = useImageUploadButton({ 
    allowMultiple: false, 
    onUpload: (imageDTOs) => {
      // Handle workflow file upload - this would need to be adapted for workflow files
    } 
  });

  const handleBrowseTemplates = useCallback(() => {
    workflowLibraryModal.open();
  }, [workflowLibraryModal]);

  const handleCreateNew = useCallback(() => {
    newWorkflow.createWithDialog();
  }, [newWorkflow]);

  const handleLoadFromFile = useCallback(() => {
    // Create file input for workflow files
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        onUploadWorkflow(file);
      }
    };
    input.click();
  }, [onUploadWorkflow]);

  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center" gap={2}>
      <Flex flexDir="column" w="full" gap={6} px={14} maxW={768} pt="20vh">
        <Heading mb={4}>Go deep with Workflows.</Heading>

        {/* Description */}
        <Text variant="subtext" fontSize="md" lineHeight="1.6" mb={2}>
          Workflows are reusable templates that automate image generation tasks, 
          allowing you to quickly perform complex operations and get consistent results.
        </Text>

        <Link 
          href="https://invoke-ai.github.io/InvokeAI/features/WORKFLOWS/" 
          isExternal 
          color="invokeBlue.400"
          fontSize="sm"
          mb={4}
        >
          {t('nodes.learnMore')} about creating workflows
        </Link>

        {/* Action Buttons */}
        <Flex flexDir="column" gap={4}>
          {/* Browse Workflow Templates - Updated copy per Devon's feedback */}
          <LaunchpadButton onClick={handleBrowseTemplates} gap={4}>
            <PiFolderOpenBold size={24} />
            <Flex flexDir="column" alignItems="flex-start" flex={1}>
              <Text fontWeight="semibold">Browse Workflow Templates</Text>
              <Text variant="subtext" fontSize="sm">
                Choose from pre-built workflows for common tasks
              </Text>
            </Flex>
          </LaunchpadButton>

          {/* Create a new Workflow */}
          <LaunchpadButton onClick={handleCreateNew} gap={4}>
            <PiFilePlusBold size={24} />
            <Flex flexDir="column" alignItems="flex-start" flex={1}>
              <Text fontWeight="semibold">Create a new Workflow</Text>
              <Text variant="subtext" fontSize="sm">
                Start a new workflow from scratch
              </Text>
            </Flex>
          </LaunchpadButton>

          {/* Load workflow from existing image or file - Updated copy per Devon's feedback */}
          <LaunchpadButton onClick={handleLoadFromFile} gap={4}>
            <PiUploadBold size={24} />
            <Flex flexDir="column" alignItems="flex-start" flex={1}>
              <Text fontWeight="semibold">Load workflow from existing image or file</Text>
              <Text variant="subtext" fontSize="sm">
                Drag or upload a workflow to start with an existing setup
              </Text>
            </Flex>
          </LaunchpadButton>
        </Flex>
      </Flex>
    </Flex>
  );
});

WorkflowsLaunchpadPanel.displayName = 'WorkflowsLaunchpadPanel';
