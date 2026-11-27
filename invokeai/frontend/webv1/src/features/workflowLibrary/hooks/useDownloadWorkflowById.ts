import { useAppDispatch } from 'app/store/storeHooks';
import { toast } from 'features/toast/toast';
import { workflowDownloaded } from 'features/workflowLibrary/store/actions';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyGetWorkflowQuery } from 'services/api/endpoints/workflows';

export const useDownloadWorkflowById = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [trigger, query] = useLazyGetWorkflowQuery();

  const toastError = useCallback(() => {
    toast({ status: 'error', description: t('nodes.downloadWorkflowError') });
  }, [t]);

  const downloadWorkflow = useCallback(
    async (workflowId: string) => {
      try {
        const { data } = await trigger(workflowId);
        if (!data) {
          toastError();
          return;
        }
        const { workflow } = data;
        const blob = new Blob([JSON.stringify(workflow, null, 2)]);
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `${workflow.name || 'My Workflow'}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        dispatch(workflowDownloaded());
      } catch {
        toastError();
      }
    },
    [dispatch, toastError, trigger]
  );

  return { downloadWorkflow, isLoading: query.isLoading };
};
