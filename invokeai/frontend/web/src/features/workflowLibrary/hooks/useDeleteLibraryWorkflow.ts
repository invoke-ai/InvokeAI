import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useDeleteWorkflowMutation, workflowsApi } from 'services/api/endpoints/workflows';

type UseDeleteLibraryWorkflowOptions = {
  onSuccess?: () => void;
  onError?: () => void;
};

type UseDeleteLibraryWorkflowReturn = {
  deleteWorkflow: (workflow_id: string) => Promise<void>;
  deleteWorkflowResult: ReturnType<typeof useDeleteWorkflowMutation>[1];
};

type UseDeleteLibraryWorkflow = (arg: UseDeleteLibraryWorkflowOptions) => UseDeleteLibraryWorkflowReturn;

export const useDeleteLibraryWorkflow: UseDeleteLibraryWorkflow = ({ onSuccess, onError }) => {
  const { t } = useTranslation();
  const [_deleteWorkflow, deleteWorkflowResult] = useDeleteWorkflowMutation();

  const deleteWorkflow = useCallback(
    async (workflow_id: string) => {
      try {
        await _deleteWorkflow(workflow_id).unwrap();
        toast({
          id: 'WORKFLOW_DELETED',
          title: t('toast.workflowDeleted'),
        });
        onSuccess && onSuccess();
      } catch {
        toast({
          id: `AUTH_ERROR_TOAST_${workflowsApi.endpoints.deleteWorkflow.name}`,
          title: t('toast.problemDeletingWorkflow'),
          status: 'error',
        });
        onError && onError();
      }
    },
    [_deleteWorkflow, t, onSuccess, onError]
  );

  return { deleteWorkflow, deleteWorkflowResult };
};
