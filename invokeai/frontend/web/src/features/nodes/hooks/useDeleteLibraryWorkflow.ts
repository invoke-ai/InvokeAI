import { useAppToaster } from 'app/components/Toaster';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useDeleteWorkflowMutation } from 'services/api/endpoints/workflows';

type UseDeleteLibraryWorkflowArg = {
  onSuccess?: () => void;
  onError?: () => void;
};

export const useDeleteLibraryWorkflow = ({
  onSuccess,
  onError,
}: UseDeleteLibraryWorkflowArg) => {
  const toaster = useAppToaster();
  const { t } = useTranslation();
  const [_deleteWorkflow, deleteWorkflowResult] = useDeleteWorkflowMutation();

  const deleteWorkflow = useCallback(
    async (workflow_id: string) => {
      try {
        await _deleteWorkflow(workflow_id).unwrap();
        toaster({
          title: t('toast.workflowDeleted'),
        });
        onSuccess && onSuccess();
      } catch {
        toaster({
          title: t('toast.problemDeletingWorkflow'),
          status: 'error',
        });
        onError && onError();
      }
    },
    [_deleteWorkflow, toaster, t, onSuccess, onError]
  );

  return { deleteWorkflow, deleteWorkflowResult };
};
