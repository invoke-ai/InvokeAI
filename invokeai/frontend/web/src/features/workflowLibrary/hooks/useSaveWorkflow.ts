import { ToastId, useToast } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useWorkflow } from 'features/nodes/hooks/useWorkflow';
import {
  workflowIDChanged,
  workflowSaved,
} from 'features/nodes/store/workflowSlice';
import { WorkflowV2 } from 'features/nodes/types/workflow';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useCreateWorkflowMutation,
  useUpdateWorkflowMutation,
} from 'services/api/endpoints/workflows';
import { O } from 'ts-toolbelt';

type UseSaveLibraryWorkflowReturn = {
  saveWorkflow: () => Promise<void>;
  isLoading: boolean;
  isError: boolean;
};

type UseSaveLibraryWorkflow = () => UseSaveLibraryWorkflowReturn;

const isWorkflowWithID = (
  workflow: WorkflowV2
): workflow is O.Required<WorkflowV2, 'id'> => Boolean(workflow.id);

export const useSaveLibraryWorkflow: UseSaveLibraryWorkflow = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const workflow = useWorkflow();
  const [updateWorkflow, updateWorkflowResult] = useUpdateWorkflowMutation();
  const [createWorkflow, createWorkflowResult] = useCreateWorkflowMutation();
  const toast = useToast();
  const toastRef = useRef<ToastId | undefined>();
  const saveWorkflow = useCallback(async () => {
    toastRef.current = toast({
      title: t('workflows.savingWorkflow'),
      status: 'loading',
      duration: null,
      isClosable: false,
    });
    try {
      if (isWorkflowWithID(workflow)) {
        await updateWorkflow(workflow).unwrap();
      } else {
        const data = await createWorkflow(workflow).unwrap();
        dispatch(workflowIDChanged(data.workflow.id));
      }
      dispatch(workflowSaved());
      toast.update(toastRef.current, {
        title: t('workflows.workflowSaved'),
        status: 'success',
        duration: 1000,
        isClosable: true,
      });
    } catch (e) {
      toast.update(toastRef.current, {
        title: t('workflows.problemSavingWorkflow'),
        status: 'error',
        duration: 1000,
        isClosable: true,
      });
    }
  }, [workflow, updateWorkflow, dispatch, toast, t, createWorkflow]);
  return {
    saveWorkflow,
    isLoading: updateWorkflowResult.isLoading || createWorkflowResult.isLoading,
    isError: updateWorkflowResult.isError || createWorkflowResult.isError,
  };
};
