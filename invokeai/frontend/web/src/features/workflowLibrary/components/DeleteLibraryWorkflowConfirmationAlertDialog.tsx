import { ConfirmationAlertDialog, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { toast } from 'features/toast/toast';
import { atom } from 'nanostores';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useDeleteWorkflowMutation, workflowsApi } from 'services/api/endpoints/workflows';
import type { WorkflowRecordListItemDTO } from 'services/api/types';

const $workflowToDelete = atom<WorkflowRecordListItemDTO | null>(null);
const clearWorkflowToDelete = () => $workflowToDelete.set(null);

export const useDeleteWorkflow = () => {
  const deleteWorkflow = useCallback((workflow: WorkflowRecordListItemDTO) => {
    $workflowToDelete.set(workflow);
  }, []);

  return deleteWorkflow;
};

export const DeleteWorkflowDialog = () => {
  useAssertSingleton('DeleteWorkflowDialog');
  const { t } = useTranslation();
  const workflowToDelete = useStore($workflowToDelete);
  const [_deleteWorkflow] = useDeleteWorkflowMutation();

  const deleteWorkflow = useCallback(async () => {
    if (!workflowToDelete) {
      return;
    }
    try {
      await _deleteWorkflow(workflowToDelete.workflow_id).unwrap();
      toast({
        id: 'WORKFLOW_DELETED',
        title: t('toast.workflowDeleted'),
      });
    } catch {
      toast({
        id: `AUTH_ERROR_TOAST_${workflowsApi.endpoints.deleteWorkflow.name}`,
        title: t('toast.problemDeletingWorkflow'),
        status: 'error',
      });
    }
  }, [_deleteWorkflow, t, workflowToDelete]);

  return (
    <ConfirmationAlertDialog
      isOpen={workflowToDelete !== null}
      onClose={clearWorkflowToDelete}
      title={t('workflows.deleteWorkflow')}
      acceptCallback={deleteWorkflow}
      acceptButtonText={t('common.delete')}
      cancelButtonText={t('common.cancel')}
      useInert={false}
    >
      <Text>{t('workflows.deleteWorkflow2')}</Text>
    </ConfirmationAlertDialog>
  );
};
