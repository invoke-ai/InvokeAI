import { useAppDispatch } from 'app/store/storeHooks';
import { workflowLoadRequested } from 'features/nodes/store/actions';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyGetImageWorkflowQuery } from 'services/api/endpoints/images';

type UseGetAndLoadEmbeddedWorkflowOptions = {
  onSuccess?: () => void;
  onError?: () => void;
};

export const useGetAndLoadEmbeddedWorkflow = (options?: UseGetAndLoadEmbeddedWorkflowOptions) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [_getAndLoadEmbeddedWorkflow, result] = useLazyGetImageWorkflowQuery();
  const getAndLoadEmbeddedWorkflow = useCallback(
    async (imageName: string) => {
      try {
        const { data } = await _getAndLoadEmbeddedWorkflow(imageName);
        if (data) {
          dispatch(workflowLoadRequested({ data, asCopy: true }));
          // No toast - the listener for this action does that after the workflow is loaded
          options?.onSuccess && options?.onSuccess();
        } else {
          toast({
            id: 'PROBLEM_RETRIEVING_WORKFLOW',
            title: t('toast.problemRetrievingWorkflow'),
            status: 'error',
          });
        }
      } catch {
        toast({
          id: 'PROBLEM_RETRIEVING_WORKFLOW',
          title: t('toast.problemRetrievingWorkflow'),
          status: 'error',
        });
        options?.onError && options?.onError();
      }
    },
    [_getAndLoadEmbeddedWorkflow, dispatch, options, t]
  );

  return [getAndLoadEmbeddedWorkflow, result] as const;
};
