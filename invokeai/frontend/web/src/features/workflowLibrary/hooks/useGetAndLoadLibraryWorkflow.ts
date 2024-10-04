import { useToast } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { workflowLoadRequested } from 'features/nodes/store/actions';
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
  const dispatch = useAppDispatch();
  const toast = useToast();
  const { t } = useTranslation();
  const [_getAndLoadWorkflow, getAndLoadWorkflowResult] = useLazyGetWorkflowQuery();
  const getAndLoadWorkflow = useCallback(
    async (workflow_id: string) => {
      try {
        const { workflow } = await _getAndLoadWorkflow(workflow_id).unwrap();
        // This action expects a stringified workflow, instead of updating the routes and services we will just stringify it here
        dispatch(workflowLoadRequested({ data: { workflow: JSON.stringify(workflow), graph: null }, asCopy: false }));
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
    [_getAndLoadWorkflow, dispatch, arg, t, toast]
  );

  return { getAndLoadWorkflow, getAndLoadWorkflowResult };
};
