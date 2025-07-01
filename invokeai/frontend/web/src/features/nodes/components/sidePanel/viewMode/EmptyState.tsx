import { Flex, Heading, Icon, Image, Link, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { LaunchpadButton } from 'features/controlLayers/components/SimpleSession/LaunchpadButton';
import { useIsWorkflowUntouched } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { workflowModeChanged } from 'features/nodes/store/workflowLibrarySlice';
import InvokeLogoSVG from 'public/assets/images/invoke-symbol-wht-lrg.svg';
import { useCallback } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiFolderOpenBold, PiPlusBold } from 'react-icons/pi';

export const EmptyState = () => {
  const isWorkflowUntouched = useIsWorkflowUntouched();

  return (
    <Flex w="full" h="full" userSelect="none" justifyContent="center">
      <Flex
        alignItems="center"
        justifyContent="center"
        borderRadius="base"
        flexDir="column"
        gap={5}
        maxW="400px"
        pt={24}
        px={4}
      >
        <Image
          src={InvokeLogoSVG}
          alt="invoke-ai-logo"
          opacity={0.2}
          mixBlendMode="overlay"
          w={16}
          h={16}
          minW={16}
          minH={16}
          userSelect="none"
        />
        {isWorkflowUntouched ? <CleanEditorContent /> : <DirtyEditorContent />}
      </Flex>
    </Flex>
  );
};

const CleanEditorContent = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const workflowLibraryModal = useWorkflowLibraryModal();

  const onClickNewWorkflow = useCallback(() => {
    dispatch(workflowModeChanged('edit'));
  }, [dispatch]);

  return (
    <>
      <Flex flexDir="column" gap={4} w="full">
        <LaunchpadButton onClick={onClickNewWorkflow} gap={4}>
          <Icon as={PiPlusBold} boxSize={6} color="base.500" />
          <Flex flexDir="column" alignItems="flex-start" gap={1}>
            <Heading size="sm">{t('nodes.newWorkflow')}</Heading>
            <Text color="base.300" fontSize="sm">
              Create a new workflow from scratch
            </Text>
          </Flex>
        </LaunchpadButton>
        <LaunchpadButton onClick={workflowLibraryModal.open} gap={4}>
          <Icon as={PiFolderOpenBold} boxSize={6} color="base.500" />
          <Flex flexDir="column" alignItems="flex-start" gap={1}>
            <Heading size="sm">{t('nodes.loadWorkflow')}</Heading>
            <Text color="base.300" fontSize="sm">
              Browse and load existing workflows
            </Text>
          </Flex>
        </LaunchpadButton>
      </Flex>
      <Text textAlign="center" fontSize="sm" color="base.400" mt={4}>
        <Trans i18nKey="nodes.workflowHelpText" size="sm" components={workflowHelpTextComponents} />
      </Text>
    </>
  );
};

const DirtyEditorContent = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    dispatch(workflowModeChanged('edit'));
  }, [dispatch]);

  return (
    <>
      <Text textAlign="center" fontSize="md" mb={4}>
        {t('nodes.noFieldsViewMode')}
      </Text>
      <Flex flexDir="column" gap={4} w="full">
        <LaunchpadButton onClick={onClick} gap={4}>
          <Icon as={PiPlusBold} boxSize={6} color="base.500" />
          <Flex flexDir="column" alignItems="flex-start" gap={1}>
            <Heading size="sm">{t('nodes.edit')}</Heading>
            <Text color="base.300" fontSize="sm">
              Switch to edit mode to build workflows
            </Text>
          </Flex>
        </LaunchpadButton>
      </Flex>
    </>
  );
};

const workflowHelpTextComponents = {
  LinkComponent: (
    <Link
      fontSize="md"
      fontWeight="semibold"
      href="https://support.invoke.ai/support/solutions/articles/151000159663-example-workflows"
      target="_blank"
    />
  ),
};
