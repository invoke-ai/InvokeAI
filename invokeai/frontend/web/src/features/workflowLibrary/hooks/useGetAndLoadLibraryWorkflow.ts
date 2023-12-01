import { useAppToaster } from 'app/components/Toaster';
import { useAppDispatch } from 'app/store/storeHooks';
import { workflowLoadRequested } from 'features/nodes/store/actions';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyGetWorkflowQuery } from 'services/api/endpoints/workflows';

type UseGetAndLoadLibraryWorkflowArg = {
  onSuccess?: () => void;
  onError?: () => void;
};

export const useGetAndLoadLibraryWorkflow = ({
  onSuccess,
  onError,
}: UseGetAndLoadLibraryWorkflowArg) => {
  const dispatch = useAppDispatch();
  const toaster = useAppToaster();
  const { t } = useTranslation();
  const [_getAndLoadWorkflow, getAndLoadWorkflowResult] =
    useLazyGetWorkflowQuery();
  const getAndLoadWorkflow = useCallback(
    async (workflow_id: string) => {
      try {
        const data = await _getAndLoadWorkflow(workflow_id).unwrap();
        dispatch(workflowLoadRequested(data.workflow));
        // No toast - the listener for this action does that after the workflow is loaded
        onSuccess && onSuccess();
      } catch {
        toaster({
          title: t('toast.problemRetrievingWorkflow'),
          status: 'error',
        });
        onError && onError();
      }
    },
    [_getAndLoadWorkflow, dispatch, onSuccess, toaster, t, onError]
  );

  return { getAndLoadWorkflow, getAndLoadWorkflowResult };
};
