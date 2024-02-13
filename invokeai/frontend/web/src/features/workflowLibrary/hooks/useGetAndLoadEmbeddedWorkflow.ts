import { useAppToaster } from 'app/components/Toaster';
import { useAppDispatch } from 'app/store/storeHooks';
import { workflowLoadRequested } from 'features/nodes/store/actions';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyGetImageWorkflowQuery } from 'services/api/endpoints/images';

type UseGetAndLoadEmbeddedWorkflowOptions = {
  onSuccess?: () => void;
  onError?: () => void;
};

type UseGetAndLoadEmbeddedWorkflowReturn = {
  getAndLoadEmbeddedWorkflow: (imageName: string) => Promise<void>;
  getAndLoadEmbeddedWorkflowResult: ReturnType<typeof useLazyGetImageWorkflowQuery>[1];
};

type UseGetAndLoadEmbeddedWorkflow = (
  options: UseGetAndLoadEmbeddedWorkflowOptions
) => UseGetAndLoadEmbeddedWorkflowReturn;

export const useGetAndLoadEmbeddedWorkflow: UseGetAndLoadEmbeddedWorkflow = ({ onSuccess, onError }) => {
  const dispatch = useAppDispatch();
  const toaster = useAppToaster();
  const { t } = useTranslation();
  const [_getAndLoadEmbeddedWorkflow, getAndLoadEmbeddedWorkflowResult] = useLazyGetImageWorkflowQuery();
  const getAndLoadEmbeddedWorkflow = useCallback(
    async (imageName: string) => {
      try {
        const workflow = await _getAndLoadEmbeddedWorkflow(imageName);
        dispatch(workflowLoadRequested({ workflow: workflow.data, asCopy: true }));
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
    [_getAndLoadEmbeddedWorkflow, dispatch, onSuccess, toaster, t, onError]
  );

  return { getAndLoadEmbeddedWorkflow, getAndLoadEmbeddedWorkflowResult };
};
