import { useAppToaster } from 'app/components/Toaster';
import { useAppDispatch } from 'app/store/storeHooks';
import { useWorkflow } from 'features/nodes/hooks/useWorkflow';
import { workflowLoaded } from 'features/nodes/store/workflowSlice';
import { zWorkflowV2 } from 'features/nodes/types/workflow';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useCreateWorkflowMutation,
  useUpdateWorkflowMutation,
} from 'services/api/endpoints/workflows';

type UseSaveLibraryWorkflowReturn = {
  saveWorkflow: () => Promise<void>;
  isLoading: boolean;
  isError: boolean;
};

type UseSaveLibraryWorkflow = () => UseSaveLibraryWorkflowReturn;

export const useSaveLibraryWorkflow: UseSaveLibraryWorkflow = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const workflow = useWorkflow();
  const [updateWorkflow, updateWorkflowResult] = useUpdateWorkflowMutation();
  const [createWorkflow, createWorkflowResult] = useCreateWorkflowMutation();
  const toaster = useAppToaster();
  const saveWorkflow = useCallback(async () => {
    try {
      if (workflow.id) {
        const data = await updateWorkflow(workflow).unwrap();
        const updatedWorkflow = zWorkflowV2.parse(data.workflow);
        dispatch(workflowLoaded(updatedWorkflow));
        toaster({
          title: t('workflows.workflowSaved'),
          status: 'success',
        });
      } else {
        const data = await createWorkflow(workflow).unwrap();
        const createdWorkflow = zWorkflowV2.parse(data.workflow);
        dispatch(workflowLoaded(createdWorkflow));
        toaster({
          title: t('workflows.workflowSaved'),
          status: 'success',
        });
      }
    } catch (e) {
      toaster({
        title: t('workflows.problemSavingWorkflow'),
        status: 'error',
      });
    }
  }, [workflow, updateWorkflow, dispatch, toaster, t, createWorkflow]);
  return {
    saveWorkflow,
    isLoading: updateWorkflowResult.isLoading || createWorkflowResult.isLoading,
    isError: updateWorkflowResult.isError || createWorkflowResult.isError,
  };
};
