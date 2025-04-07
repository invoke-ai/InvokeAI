import { Button, Flex, Heading, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowId } from 'features/nodes/store/selectors';
import { toast } from 'features/toast/toast';
import { useSaveOrSaveAsWorkflow } from 'features/workflowLibrary/hooks/useSaveOrSaveAsWorkflow';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold, PiLockOpenBold } from 'react-icons/pi';
import { useUnpublishWorkflowMutation } from 'services/api/endpoints/workflows';

export const PublishedWorkflowPanelContent = memo(() => {
  const { t } = useTranslation();
  const saveAs = useSaveOrSaveAsWorkflow();
  const [unpublishWorkflow] = useUnpublishWorkflowMutation();
  const workflowId = useAppSelector(selectWorkflowId);

  const handleUnpublish = useCallback(async () => {
    if (workflowId) {
      try {
        await unpublishWorkflow(workflowId).unwrap();
        toast({
          title: t('toast.workflowUnpublished'),
          status: 'success',
        });
      } catch (error) {
        toast({
          title: t('toast.problemUnpublishingWorkflow'),
          description: t('toast.problemUnpublishingWorkflowDescription'),
          status: 'error',
        });
      }
    }
  }, [unpublishWorkflow, workflowId, t]);

  return (
    <Flex flexDir="column" w="full" h="full" gap={2} alignItems="center">
      <Heading size="md" pt={32}>
        {t('workflows.builder.workflowLocked')}
      </Heading>
      <Text fontSize="md">{t('workflows.builder.publishedWorkflowsLocked')}</Text>
      <Button size="md" onClick={saveAs} variant="ghost" leftIcon={<PiCopyBold />}>
        {t('common.saveAs')}
      </Button>
      <Button size="md" onClick={handleUnpublish} variant="ghost" leftIcon={<PiLockOpenBold />}>
        {t('workflows.builder.unpublish')}
      </Button>
    </Flex>
  );
});
PublishedWorkflowPanelContent.displayName = 'PublishedWorkflowPanelContent';
