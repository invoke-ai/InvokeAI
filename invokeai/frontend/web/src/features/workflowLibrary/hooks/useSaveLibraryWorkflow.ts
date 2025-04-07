import type { ToastId } from '@invoke-ai/ui-library';
import { useToast } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { formFieldInitialValuesChanged } from 'features/nodes/store/nodesSlice';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { useGetFormFieldInitialValues } from 'features/workflowLibrary/hooks/useGetFormInitialValues';
import { workflowUpdated } from 'features/workflowLibrary/store/actions';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useUpdateWorkflowMutation, workflowsApi } from 'services/api/endpoints/workflows';
import type { SetFieldType, SetRequired } from 'type-fest';

/**
 * A library workflow is a workflow that is already saved in the library. It has an id and is not in the default category.
 */
type LibraryWorkflow = SetFieldType<
  SetRequired<WorkflowV3, 'id'>,
  'meta',
  SetFieldType<WorkflowV3['meta'], 'category', Exclude<WorkflowV3['meta']['category'], 'default'>>
>;

export const isLibraryWorkflow = (workflow: WorkflowV3): workflow is LibraryWorkflow =>
  !!workflow.id && workflow.meta.category !== 'default';

type UseSaveLibraryWorkflowReturn = {
  saveWorkflow: (workflow: LibraryWorkflow) => Promise<void>;
  isLoading: boolean;
  isError: boolean;
};

export const useSaveLibraryWorkflow = (): UseSaveLibraryWorkflowReturn => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const getFormFieldInitialValues = useGetFormFieldInitialValues();
  const [updateWorkflow, { isLoading, isError }] = useUpdateWorkflowMutation();
  const toast = useToast();
  const toastRef = useRef<ToastId | undefined>();

  const saveWorkflow = useCallback(
    async (workflow: LibraryWorkflow) => {
      toastRef.current = toast({
        title: t('workflows.savingWorkflow'),
        status: 'loading',
        duration: null,
        isClosable: false,
      });
      try {
        await updateWorkflow(workflow).unwrap();
        dispatch(workflowUpdated());
        // When a workflow is saved, the form field initial values are updated to the current form field values
        dispatch(formFieldInitialValuesChanged({ formFieldInitialValues: getFormFieldInitialValues() }));
        toast.update(toastRef.current, {
          title: t('workflows.workflowSaved'),
          status: 'success',
          duration: 1000,
          isClosable: true,
        });
      } catch (e) {
        if (!toast.isActive(`auth-error-toast-${workflowsApi.endpoints.updateWorkflow.name}`)) {
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
    [toast, t, dispatch, getFormFieldInitialValues, updateWorkflow]
  );
  return {
    saveWorkflow,
    isLoading,
    isError,
  };
};
