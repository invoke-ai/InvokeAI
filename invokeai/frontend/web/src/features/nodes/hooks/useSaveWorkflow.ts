import { useAppToaster } from 'app/components/Toaster';
import { useAppDispatch } from 'app/store/storeHooks';
import { useWorkflow } from 'features/nodes/hooks/useWorkflow';
import { workflowLoaded } from 'features/nodes/store/nodesSlice';
import { zWorkflowV2 } from 'features/nodes/types/workflow';
import { useCallback } from 'react';
import {
  useCreateWorkflowMutation,
  useUpdateWorkflowMutation,
} from 'services/api/endpoints/workflows';

export const useSaveWorkflow = () => {
  const dispatch = useAppDispatch();
  const workflow = useWorkflow();
  const [updateWorkflow, updateWorkflowResult] = useUpdateWorkflowMutation();
  const [createWorkflow, createWorkflowResult] = useCreateWorkflowMutation();
  const toaster = useAppToaster();
  const saveWorkflow = useCallback(async () => {
    try {
      if (workflow.id) {
        console.log('update workflow');
        const data = await updateWorkflow(workflow).unwrap();
        const updatedWorkflow = zWorkflowV2.parse(data.workflow);
        dispatch(workflowLoaded(updatedWorkflow));
      } else {
        console.log('create workflow');
        const data = await createWorkflow(workflow).unwrap();
        const createdWorkflow = zWorkflowV2.parse(data.workflow);
        dispatch(workflowLoaded(createdWorkflow));
      }
      toaster({
        title: 'Workflow saved',
        status: 'success',
        duration: 3000,
      });
    } catch (e) {
      toaster({
        title: 'Failed to save workflow',
        // description: e.message,
        status: 'error',
        duration: 3000,
      });
    }
  }, [workflow, toaster, updateWorkflow, dispatch, createWorkflow]);
  return saveWorkflow;
};
