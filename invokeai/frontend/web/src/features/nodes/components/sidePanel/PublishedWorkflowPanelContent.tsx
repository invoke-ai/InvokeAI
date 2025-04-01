import { Button, Flex, Heading, Text } from '@invoke-ai/ui-library';
import { useSaveOrSaveAsWorkflow } from 'features/workflowLibrary/hooks/useSaveOrSaveAsWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold, PiLockOpenBold } from 'react-icons/pi';

export const PublishedWorkflowPanelContent = memo(() => {
  const { t } = useTranslation();
  const saveAs = useSaveOrSaveAsWorkflow();
  return (
    <Flex flexDir="column" w="full" h="full" gap={2} alignItems="center">
      <Heading size="md" pt={32}>
        {t('workflows.builder.workflowLocked')}
      </Heading>
      <Text fontSize="md">{t('workflows.builder.publishedWorkflowsLocked')}</Text>
      <Button size="md" onClick={saveAs} variant="ghost" leftIcon={<PiCopyBold />}>
        {t('common.saveAs')}
      </Button>
      <Button size="md" onClick={undefined} variant="ghost" leftIcon={<PiLockOpenBold />}>
        {t('workflows.builder.unpublish')}
      </Button>
    </Flex>
  );
});
PublishedWorkflowPanelContent.displayName = 'PublishedWorkflowPanelContent';
