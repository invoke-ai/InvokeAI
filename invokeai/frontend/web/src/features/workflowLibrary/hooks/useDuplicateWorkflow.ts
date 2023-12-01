import { useAppToaster } from 'app/components/Toaster';
import { useAppDispatch } from 'app/store/storeHooks';
import { useWorkflow } from 'features/nodes/hooks/useWorkflow';
import { workflowLoaded } from 'features/nodes/store/nodesSlice';
import { zWorkflowV2 } from 'features/nodes/types/workflow';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCreateWorkflowMutation } from 'services/api/endpoints/workflows';

type Arg = {
  name: string;
  onSuccess?: () => void;
  onError?: () => void;
};

export const useSaveWorkflowAs = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const workflow = useWorkflow();
  const [createWorkflow, createWorkflowResult] = useCreateWorkflowMutation();
  const toaster = useAppToaster();
  const saveWorkflowAs = useCallback(
    async ({ name: newName, onSuccess, onError }: Arg) => {
      try {
        workflow.id = undefined;
        workflow.name = newName;
        const data = await createWorkflow(workflow).unwrap();
        const createdWorkflow = zWorkflowV2.parse(data.workflow);
        dispatch(workflowLoaded(createdWorkflow));
        onSuccess && onSuccess();
        toaster({
          title: t('workflows.workflowSaved'),
          status: 'success',
        });
      } catch (e) {
        onError && onError();
        toaster({
          title: t('workflows.problemSavingWorkflow'),
          status: 'error',
        });
      }
    },
    [workflow, dispatch, toaster, t, createWorkflow]
  );
  return {
    saveWorkflowAs,
    isLoading: createWorkflowResult.isLoading,
    isError: createWorkflowResult.isError,
  };
};
