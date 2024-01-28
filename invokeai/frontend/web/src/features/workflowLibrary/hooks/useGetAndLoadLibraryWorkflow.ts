import { useToast } from '@invoke-ai/ui-library';
import { useAppToaster } from 'app/components/Toaster';
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

type UseGetAndLoadLibraryWorkflow = (arg: UseGetAndLoadLibraryWorkflowOptions) => UseGetAndLoadLibraryWorkflowReturn;

export const useGetAndLoadLibraryWorkflow: UseGetAndLoadLibraryWorkflow = ({ onSuccess, onError }) => {
  const dispatch = useAppDispatch();
  const toaster = useAppToaster();
  const toast = useToast();
  const { t } = useTranslation();
  const [_getAndLoadWorkflow, getAndLoadWorkflowResult] = useLazyGetWorkflowQuery();
  const getAndLoadWorkflow = useCallback(
    async (workflow_id: string) => {
      try {
        const data = await _getAndLoadWorkflow(workflow_id).unwrap();
        dispatch(workflowLoadRequested({ workflow: data.workflow, asCopy: false }));
        // No toast - the listener for this action does that after the workflow is loaded
        onSuccess && onSuccess();
      } catch {
        if (!toast.isActive(`auth-error-toast-${workflowsApi.endpoints.getWorkflow.name}`)) {
          toaster({
            title: t('toast.problemRetrievingWorkflow'),
            status: 'error',
          });
        }
        onError && onError();
      }
    },
    [_getAndLoadWorkflow, dispatch, onSuccess, toaster, t, onError, toast]
  );

  return { getAndLoadWorkflow, getAndLoadWorkflowResult };
};
