import { useAppToaster } from 'app/components/Toaster';
import { useAppDispatch } from 'app/store/storeHooks';
import { useWorkflow } from 'features/nodes/hooks/useWorkflow';
import { workflowLoaded } from 'features/nodes/store/nodesSlice';
import { zWorkflowV2 } from 'features/nodes/types/workflow';
import { omit } from 'lodash-es';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCreateWorkflowMutation } from 'services/api/endpoints/workflows';

export const useDuplicateLibraryWorkflow = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const workflow = useWorkflow();
  const [createWorkflow, createWorkflowResult] = useCreateWorkflowMutation();
  const toaster = useAppToaster();
  const duplicateWorkflow = useCallback(async () => {
    try {
      const data = await createWorkflow(omit(workflow, 'id')).unwrap();
      const createdWorkflow = zWorkflowV2.parse(data.workflow);
      dispatch(workflowLoaded(createdWorkflow));
      toaster({
        title: t('workflows.workflowSaved'),
        status: 'success',
      });
    } catch (e) {
      toaster({
        title: t('workflows.problemSavingWorkflow'),
        status: 'error',
      });
    }
  }, [workflow, dispatch, toaster, t, createWorkflow]);
  return {
    duplicateWorkflow,
    isLoading: createWorkflowResult.isLoading,
    isError: createWorkflowResult.isError,
  };
};
