import { Button, Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
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
          {t('ui.launchpad.workflows.description')}
        </Text>

        <Text>
          <Button
            as="a"
            variant="link"
            href="https://support.invoke.ai/support/solutions/articles/151000189610-getting-started-with-workflows-denoise-latents"
            target="_blank"
            rel="noopener noreferrer"
            size="sm"
          >
            {t('ui.launchpad.workflows.learnMoreLink')}
          </Button>
        </Text>

        {/* Action Buttons */}
        <Flex flexDir="column" gap={8}>
          {/* Browse Workflow Templates */}
          <LaunchpadButton onClick={handleBrowseTemplates} position="relative" gap={8}>
            <Icon as={PiFolderOpenBold} boxSize={8} color="base.500" />
            <Flex flexDir="column" alignItems="flex-start" gap={2}>
              <Heading size="sm">{t('ui.launchpad.workflows.browseTemplates.title')}</Heading>
              <Text color="base.300">{t('ui.launchpad.workflows.browseTemplates.description')}</Text>
            </Flex>
          </LaunchpadButton>

          {/* Create a new Workflow */}
          <LaunchpadButton onClick={handleCreateNew} position="relative" gap={8}>
            <Icon as={PiFilePlusBold} boxSize={8} color="base.500" />
            <Flex flexDir="column" alignItems="flex-start" gap={2}>
              <Heading size="sm">{t('ui.launchpad.workflows.createNew.title')}</Heading>
              <Text color="base.300">{t('ui.launchpad.workflows.createNew.description')}</Text>
            </Flex>
          </LaunchpadButton>

          {/* Load workflow from existing image or file */}
          <LaunchpadButton onClick={handleLoadFromFile} position="relative" gap={8}>
            <Icon as={PiUploadBold} boxSize={8} color="base.500" />
            <Flex flexDir="column" alignItems="flex-start" gap={2}>
              <Heading size="sm">{t('ui.launchpad.workflows.loadFromFile.title')}</Heading>
              <Text color="base.300">{t('ui.launchpad.workflows.loadFromFile.description')}</Text>
            </Flex>
          </LaunchpadButton>
        </Flex>
      </Flex>
    </Flex>
  );
});

WorkflowsLaunchpadPanel.displayName = 'WorkflowsLaunchpadPanel';
