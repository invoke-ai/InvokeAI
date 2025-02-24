import type { ToastId } from '@invoke-ai/ui-library';
import { useToast } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { $builtWorkflow } from 'features/nodes/hooks/useWorkflowWatcher';
import { formFieldInitialValuesChanged, workflowIDChanged, workflowSaved } from 'features/nodes/store/workflowSlice';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { useGetFormFieldInitialValues } from 'features/workflowLibrary/hooks/useGetFormInitialValues';
import { workflowUpdated } from 'features/workflowLibrary/store/actions';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useCreateWorkflowMutation, useUpdateWorkflowMutation, workflowsApi } from 'services/api/endpoints/workflows';
import type { SetRequired } from 'type-fest';

type UseSaveLibraryWorkflowReturn = {
  saveWorkflow: () => Promise<void>;
  isLoading: boolean;
  isError: boolean;
};

type UseSaveLibraryWorkflow = () => UseSaveLibraryWorkflowReturn;

export const isWorkflowWithID = (workflow: WorkflowV3): workflow is SetRequired<WorkflowV3, 'id'> =>
  Boolean(workflow.id);

export const useSaveLibraryWorkflow: UseSaveLibraryWorkflow = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const getFormFieldInitialValues = useGetFormFieldInitialValues();
  const [updateWorkflow, updateWorkflowResult] = useUpdateWorkflowMutation();
  const [createWorkflow, createWorkflowResult] = useCreateWorkflowMutation();
  const toast = useToast();
  const toastRef = useRef<ToastId | undefined>();
  const saveWorkflow = useCallback(async () => {
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
      if (isWorkflowWithID(workflow)) {
        await updateWorkflow(workflow).unwrap();
        dispatch(workflowUpdated());
      } else {
        const data = await createWorkflow(workflow).unwrap();
        dispatch(workflowIDChanged(data.workflow.id));
      }
      dispatch(workflowSaved());
      // When a workflow is saved, the form field initial values are updated to the current form field values
      dispatch(formFieldInitialValuesChanged({ formFieldInitialValues: getFormFieldInitialValues() }));
      toast.update(toastRef.current, {
        title: t('workflows.workflowSaved'),
        status: 'success',
        duration: 1000,
        isClosable: true,
      });
    } catch (e) {
      if (
        !toast.isActive(`auth-error-toast-${workflowsApi.endpoints.createWorkflow.name}`) &&
        !toast.isActive(`auth-error-toast-${workflowsApi.endpoints.updateWorkflow.name}`)
      ) {
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
  }, [toast, t, dispatch, getFormFieldInitialValues, updateWorkflow, createWorkflow]);
  return {
    saveWorkflow,
    isLoading: updateWorkflowResult.isLoading || createWorkflowResult.isLoading,
    isError: updateWorkflowResult.isError || createWorkflowResult.isError,
  };
};
