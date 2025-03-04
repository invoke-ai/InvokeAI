import { useToast } from '@invoke-ai/ui-library';
import { useLoadWorkflow } from 'features/workflowLibrary/hooks/useLoadWorkflow';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyGetWorkflowQuery, workflowsApi } from 'services/api/endpoints/workflows';

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
  const [_getAndLoadWorkflow, getAndLoadWorkflowResult] = useLazyGetWorkflowQuery();
  const getAndLoadWorkflow = useCallback(
    async (workflow_id: string) => {
      try {
        const { workflow } = await _getAndLoadWorkflow(workflow_id).unwrap();
        // This action expects a stringified workflow, instead of updating the routes and services we will just stringify it here
        loadWorkflow({ workflow: JSON.stringify(workflow), graph: null });
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
    [_getAndLoadWorkflow, loadWorkflow, arg, toast, t]
  );

  return { getAndLoadWorkflow, getAndLoadWorkflowResult };
};
