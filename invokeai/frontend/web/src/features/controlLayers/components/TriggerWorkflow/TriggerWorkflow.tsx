import { Button, ButtonGroup, Flex, Heading, Spacer, Spinner, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useWorkflowTriggerApply } from 'features/controlLayers/hooks/useWorkflowTriggerApply';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { parseAndMigrateWorkflow } from 'features/nodes/util/workflow/migrations';
import { toast } from 'features/toast/toast';
import {
  setWorkflowLibraryBrowseIntent,
  setWorkflowLibraryTriggerIntent,
} from 'features/workflowLibrary/store/workflowLibraryIntent';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBooksDuotone, PiPlayBold, PiXBold } from 'react-icons/pi';
import { useLazyGetWorkflowQuery } from 'services/api/endpoints/workflows';
import type { S } from 'services/api/types';

const TriggerWorkflowContent = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const workflowLibraryModal = useWorkflowLibraryModal();
  const [fetchWorkflow, { isFetching: isFetchingWorkflow }] = useLazyGetWorkflowQuery();
  const isBusy = useCanvasIsBusy();
  const { apply, isApplying, selectedWorkflow, selectedWorkflowName } = useWorkflowTriggerApply();

  const onSelectWorkflow = useCallback(
    (workflow: S['WorkflowRecordListItemWithThumbnailDTO']) => {
      const handleSelection = async () => {
        try {
          const res = await fetchWorkflow(workflow.workflow_id).unwrap();
          const migratedWorkflow = parseAndMigrateWorkflow(res.workflow);
          canvasManager.stateApi.setWorkflowTriggerSelection({
            workflow: migratedWorkflow,
            workflowId: workflow.workflow_id,
            workflowName: migratedWorkflow.name ?? workflow.name,
          });
        } catch {
          toast({
            status: 'error',
            title: t('workflows.problemRetrievingWorkflow'),
          });
        } finally {
          setWorkflowLibraryBrowseIntent();
        }
      };

      void handleSelection();
    },
    [canvasManager.stateApi, fetchWorkflow, t]
  );

  const openWorkflowLibrary = useCallback(() => {
    setWorkflowLibraryTriggerIntent((workflow) => {
      onSelectWorkflow(workflow);
      workflowLibraryModal.close();
    });
    workflowLibraryModal.open();
  }, [onSelectWorkflow, workflowLibraryModal]);

  const isApplyDisabled = useMemo(() => {
    return !selectedWorkflow || isApplying || isFetchingWorkflow || isBusy;
  }, [selectedWorkflow, isApplying, isFetchingWorkflow, isBusy]);

  const cancel = useCallback(() => {
    setWorkflowLibraryBrowseIntent();
    canvasManager.stateApi.cancelWorkflowTrigger();
  }, [canvasManager.stateApi]);

  return (
    <Flex bg="base.800" borderRadius="base" p={4} flexDir="column" gap={4} w={420} shadow="dark-lg">
      <Flex w="full" gap={4} alignItems="center">
        <Heading size="md" color="base.300" userSelect="none">
          {t('controlLayers.triggerWorkflow.heading')}
        </Heading>
        <Spacer />
      </Flex>
      <Flex flexDir="column" gap={2}>
        <Text color="base.400">{selectedWorkflowName ?? t('controlLayers.triggerWorkflow.noWorkflowSelected')}</Text>
        {isFetchingWorkflow && (
          <Flex alignItems="center" gap={2} color="base.500">
            <Spinner size="sm" />
            <Text>{t('controlLayers.triggerWorkflow.loading')}</Text>
          </Flex>
        )}
      </Flex>
      <ButtonGroup isAttached={false} size="sm" w="full">
        <Button
          variant="ghost"
          leftIcon={<PiBooksDuotone />}
          onClick={openWorkflowLibrary}
          isDisabled={isApplying || isFetchingWorkflow || isBusy}
        >
          {t('controlLayers.triggerWorkflow.openLibrary')}
        </Button>
        <Spacer />
        <Button variant="ghost" leftIcon={<PiPlayBold />} onClick={apply} isDisabled={isApplyDisabled}>
          {isApplying ? t('controlLayers.triggerWorkflow.applying') : t('controlLayers.triggerWorkflow.apply')}
          {isApplying && <Spinner size="sm" ml={2} />}
        </Button>
        <Button variant="ghost" leftIcon={<PiXBold />} onClick={cancel} isDisabled={isApplying}>
          {t('controlLayers.triggerWorkflow.cancel')}
        </Button>
      </ButtonGroup>
    </Flex>
  );
});

TriggerWorkflowContent.displayName = 'TriggerWorkflowContent';

export const TriggerWorkflow = () => {
  const canvasManager = useCanvasManager();
  const state = useStore(canvasManager.stateApi.$workflowTrigger);
  if (!state) {
    return null;
  }
  return <TriggerWorkflowContent />;
};

TriggerWorkflow.displayName = 'TriggerWorkflow';
