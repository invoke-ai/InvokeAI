import { ToastId, useToast } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useWorkflow } from 'features/nodes/hooks/useWorkflow';
import { workflowLoaded } from 'features/nodes/store/actions';
import { zWorkflowV2 } from 'features/nodes/types/workflow';
import { useCallback, useRef } from 'react';
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
      if (workflow.id) {
        const data = await updateWorkflow(workflow).unwrap();
        const updatedWorkflow = zWorkflowV2.parse(data.workflow);
        dispatch(workflowLoaded(updatedWorkflow));
      } else {
        const data = await createWorkflow(workflow).unwrap();
        const createdWorkflow = zWorkflowV2.parse(data.workflow);
        dispatch(workflowLoaded(createdWorkflow));
      }
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
