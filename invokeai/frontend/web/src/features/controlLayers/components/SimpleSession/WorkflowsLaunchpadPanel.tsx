import { Button, Flex, Heading, Icon, Link, Text } from '@invoke-ai/ui-library';
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
      <Flex flexDir="column" w="full" gap={4} px={14} maxW={768} pt="20vh">
        <Heading mb={4}>{t('ui.launchpad.workflowsTitle')}</Heading>

        {/* Description */}
        <Text variant="subtext" fontSize="md" lineHeight="1.6">
          Workflows are reusable templates that automate image generation tasks, 
          allowing you to quickly perform complex operations and get consistent results.
        </Text>

        <Link 
          href="https://invoke-ai.github.io/InvokeAI/features/WORKFLOWS/" 
          isExternal 
          color="invokeBlue.400"
          fontSize="sm"
        >
          {t('learnMore')} about creating workflows
        </Link>

        {/* Action Buttons */}
        <Flex flexDir="column" gap={8}>
          {/* Browse Workflow Templates */}
          <LaunchpadButton onClick={handleBrowseTemplates} position="relative" gap={8}>
            <Icon as={PiFolderOpenBold} boxSize={8} color="base.500" />
            <Flex flexDir="column" alignItems="flex-start" gap={2}>
              <Heading size="sm">Browse Workflow Templates</Heading>
              <Text color="base.300">Choose from pre-built workflows for common tasks</Text>
            </Flex>
          </LaunchpadButton>

          {/* Create a new Workflow */}
          <LaunchpadButton onClick={handleCreateNew} position="relative" gap={8}>
            <Icon as={PiFilePlusBold} boxSize={8} color="base.500" />
            <Flex flexDir="column" alignItems="flex-start" gap={2}>
              <Heading size="sm">Create a new Workflow</Heading>
              <Text color="base.300">Start a new workflow from scratch</Text>
            </Flex>
          </LaunchpadButton>

          {/* Load workflow from existing image or file */}
          <LaunchpadButton onClick={handleLoadFromFile} position="relative" gap={8}>
            <Icon as={PiUploadBold} boxSize={8} color="base.500" />
            <Flex flexDir="column" alignItems="flex-start" gap={2}>
              <Heading size="sm">Load workflow from existing image or file</Heading>
              <Text color="base.300">Drag or upload a workflow to start with an existing setup</Text>
            </Flex>
          </LaunchpadButton>
        </Flex>
      </Flex>
    </Flex>
  );
});

WorkflowsLaunchpadPanel.displayName = 'WorkflowsLaunchpadPanel';
