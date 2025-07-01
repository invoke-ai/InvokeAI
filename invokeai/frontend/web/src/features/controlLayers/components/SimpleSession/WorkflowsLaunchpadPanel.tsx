import { Button, Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { useNewWorkflow } from 'features/workflowLibrary/components/NewWorkflowConfirmationAlertDialog';
import { memo, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiFilePlusBold, PiFolderOpenBold, PiUploadBold } from 'react-icons/pi';

import { LaunchpadButton } from './LaunchpadButton';

export const WorkflowsLaunchpadPanel = memo(() => {
  const { t } = useTranslation();
  const workflowLibraryModal = useWorkflowLibraryModal();
  const newWorkflow = useNewWorkflow();

  const handleBrowseTemplates = useCallback(() => {
    workflowLibraryModal.open();
  }, [workflowLibraryModal]);

  const handleCreateNew = useCallback(() => {
    newWorkflow.createWithDialog();
  }, [newWorkflow]);

  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();

  const onDropAccepted = useCallback(
    ([file]: File[]) => {
      if (!file) {
        return;
      }
      loadWorkflowWithDialog({
        type: 'file',
        data: file,
      });
    },
    [loadWorkflowWithDialog]
  );

  const uploadApi = useDropzone({
    accept: { 'application/json': ['.json'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });

  return (
    <FocusRegionWrapper region="launchpad" as={Flex} flexDir="column" h="full" w="full" alignItems="center" gap={2}>
      <Flex flexDir="column" w="full" gap={4} px={14} maxW={768} pt="20vh">
        <Heading>{t('ui.launchpad.workflowsTitle')}</Heading>

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
          <LaunchpadButton {...uploadApi.getRootProps()} position="relative" gap={8}>
            <Icon as={PiUploadBold} boxSize={8} color="base.500" />
            <Flex flexDir="column" alignItems="flex-start" gap={2}>
              <Heading size="sm">{t('ui.launchpad.workflows.loadFromFile.title')}</Heading>
              <Text color="base.300">{t('ui.launchpad.workflows.loadFromFile.description')}</Text>
            </Flex>
            <Flex position="absolute" right={3} bottom={3}>
              <PiUploadBold />
              <input {...uploadApi.getInputProps()} />
            </Flex>
          </LaunchpadButton>
        </Flex>
      </Flex>
    </FocusRegionWrapper>
  );
});

WorkflowsLaunchpadPanel.displayName = 'WorkflowsLaunchpadPanel';
