import { Flex, Heading, Icon, Link, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useIsWorkflowUntouched } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { workflowModeChanged } from 'features/nodes/store/workflowLibrarySlice';
import { LaunchpadButton } from 'features/ui/layouts/LaunchpadButton';
import { useCallback } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiFolderOpenBold, PiPlusBold } from 'react-icons/pi';

export const EmptyState = () => {
  const isWorkflowUntouched = useIsWorkflowUntouched();

  if (isWorkflowUntouched) {
    return <CleanEditorContent />;
  }

  return <DirtyEditorContent />;
};

const CleanEditorContent = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const workflowLibraryModal = useWorkflowLibraryModal();

  const onClickNewWorkflow = useCallback(() => {
    dispatch(workflowModeChanged('edit'));
  }, [dispatch]);

  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center">
      <Flex flexDir="column" gap={8} w="full" pt="20vh" px={8} maxW={768}>
        <LaunchpadButton onClick={onClickNewWorkflow} gap={8}>
          <Icon as={PiPlusBold} boxSize={6} color="base.500" />
          <Flex flexDir="column" alignItems="flex-start" gap={2}>
            <Heading size="sm">{t('nodes.newWorkflow')}</Heading>
            <Text color="base.300" fontSize="sm">
              {t('ui.launchpad.createNewWorkflowFromScratch')}
            </Text>
          </Flex>
        </LaunchpadButton>
        <LaunchpadButton onClick={workflowLibraryModal.open} gap={8}>
          <Icon as={PiFolderOpenBold} boxSize={6} color="base.500" />
          <Flex flexDir="column" alignItems="flex-start" gap={2}>
            <Heading size="sm">{t('nodes.loadWorkflow')}</Heading>
            <Text color="base.300" fontSize="sm">
              {t('ui.launchpad.browseAndLoadWorkflows')}
            </Text>
          </Flex>
        </LaunchpadButton>
        <Text textAlign="center" fontSize="sm" color="base.400">
          <Trans i18nKey="nodes.workflowHelpText" size="sm" components={workflowHelpTextComponents} />
        </Text>
      </Flex>
    </Flex>
  );
};

const DirtyEditorContent = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    dispatch(workflowModeChanged('edit'));
  }, [dispatch]);

  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center">
      <Flex flexDir="column" gap={8} w="full" pt="20vh" px={8} maxW={768}>
        <Text textAlign="center" fontSize="sm" color="base.300">
          {t('nodes.noFieldsViewMode')}
        </Text>
        <LaunchpadButton onClick={onClick} gap={8}>
          <Icon as={PiPlusBold} boxSize={6} color="base.500" />
          <Flex flexDir="column" alignItems="flex-start" gap={2}>
            <Heading size="sm">{t('nodes.edit')}</Heading>
            <Text color="base.300" fontSize="sm">
              Switch to edit mode to build workflows
            </Text>
          </Flex>
        </LaunchpadButton>
      </Flex>
    </Flex>
  );
};

const workflowHelpTextComponents = {
  LinkComponent: (
    <Link
      fontSize="sm"
      fontWeight="semibold"
      href="https://support.invoke.ai/support/solutions/articles/151000159663-example-workflows"
      target="_blank"
    />
  ),
};
