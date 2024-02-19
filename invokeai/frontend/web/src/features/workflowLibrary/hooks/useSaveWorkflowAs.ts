import type { ToastId } from '@invoke-ai/ui-library';
import { useToast } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { $builtWorkflow } from 'features/nodes/hooks/useWorkflowWatcher';
import {
  workflowCategoryChanged,
  workflowIDChanged,
  workflowNameChanged,
  workflowSaved,
} from 'features/nodes/store/workflowSlice';
import type { WorkflowCategory } from 'features/nodes/types/workflow';
import { newWorkflowSaved } from 'features/workflowLibrary/store/actions';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useCreateWorkflowMutation, workflowsApi } from 'services/api/endpoints/workflows';

type SaveWorkflowAsArg = {
  name: string;
  category: WorkflowCategory;
  onSuccess?: () => void;
  onError?: () => void;
};

type UseSaveWorkflowAsReturn = {
  saveWorkflowAs: (arg: SaveWorkflowAsArg) => Promise<void>;
  isLoading: boolean;
  isError: boolean;
};

type UseSaveWorkflowAs = () => UseSaveWorkflowAsReturn;

export const useSaveWorkflowAs: UseSaveWorkflowAs = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [createWorkflow, createWorkflowResult] = useCreateWorkflowMutation();
  const toast = useToast();
  const toastRef = useRef<ToastId | undefined>();
  const saveWorkflowAs = useCallback(
    async ({ name: newName, category, onSuccess, onError }: SaveWorkflowAsArg) => {
      const workflow = $builtWorkflow.get();
      if (!workflow) {
        return;
      }
      toastRef.current = toast({
        title: t('workflows.savingWorkflow'),
        status: 'loading',
        duration: null,
        isClosable: false,
      });
      try {
        workflow.id = undefined;
        workflow.name = newName;
        workflow.meta.category = category;

        const data = await createWorkflow(workflow).unwrap();
        dispatch(workflowIDChanged(data.workflow.id));
        dispatch(workflowNameChanged(data.workflow.name));
        dispatch(workflowCategoryChanged(data.workflow.meta.category));
        dispatch(workflowSaved());
        dispatch(newWorkflowSaved({ category }));

        onSuccess && onSuccess();
        toast.update(toastRef.current, {
          title: t('workflows.workflowSaved'),
          status: 'success',
          duration: 1000,
          isClosable: true,
        });
      } catch (e) {
        onError && onError();
        if (!toast.isActive(`auth-error-toast-${workflowsApi.endpoints.createWorkflow.name}`)) {
          toast.update(toastRef.current, {
            title: t('workflows.problemSavingWorkflow'),
            status: 'error',
            duration: 1000,
            isClosable: true,
          });
        } else {
          toast.close(toastRef.current);
        }
      }
    },
    [toast, createWorkflow, dispatch, t]
  );
  return {
    saveWorkflowAs,
    isLoading: createWorkflowResult.isLoading,
    isError: createWorkflowResult.isError,
  };
};
