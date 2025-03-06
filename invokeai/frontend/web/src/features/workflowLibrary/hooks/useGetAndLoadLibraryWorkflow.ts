import { useToast } from '@invoke-ai/ui-library';
import { useLoadWorkflow } from 'features/workflowLibrary/hooks/useLoadWorkflow';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyGetWorkflowQuery, useUpdateOpenedAtMutation, workflowsApi } from 'services/api/endpoints/workflows';

type UseGetAndLoadLibraryWorkflowOptions = {
  onSuccess?: () => void;
  onError?: () => void;
};

type UseGetAndLoadLibraryWorkflowReturn = {
  getAndLoadWorkflow: (workflow_id: string) => Promise<void>;
  getAndLoadWorkflowResult: ReturnType<typeof useLazyGetWorkflowQuery>[1];
};

type UseGetAndLoadLibraryWorkflow = (arg?: UseGetAndLoadLibraryWorkflowOptions) => UseGetAndLoadLibraryWorkflowReturn;

export const useGetAndLoadLibraryWorkflow: UseGetAndLoadLibraryWorkflow = (arg) => {
  const toast = useToast();
  const { t } = useTranslation();
  const loadWorkflow = useLoadWorkflow();
  const [getWorkflow, getAndLoadWorkflowResult] = useLazyGetWorkflowQuery();
  const [updateOpenedAt] = useUpdateOpenedAtMutation();
  const getAndLoadWorkflow = useCallback(
    async (workflow_id: string) => {
      try {
        const { workflow } = await getWorkflow(workflow_id).unwrap();
        // This action expects a stringified workflow, instead of updating the routes and services we will just stringify it here
        await loadWorkflow({ workflow: JSON.stringify(workflow), graph: null });
        updateOpenedAt({ workflow_id });
        // No toast - the listener for this action does that after the workflow is loaded
        arg?.onSuccess && arg.onSuccess();
      } catch {
        toast({
          id: `AUTH_ERROR_TOAST_${workflowsApi.endpoints.getWorkflow.name}`,
          title: t('toast.problemRetrievingWorkflow'),
          status: 'error',
        });
        arg?.onError && arg.onError();
      }
    },
    [getWorkflow, loadWorkflow, updateOpenedAt, arg, toast, t]
  );

  return { getAndLoadWorkflow, getAndLoadWorkflowResult };
};
