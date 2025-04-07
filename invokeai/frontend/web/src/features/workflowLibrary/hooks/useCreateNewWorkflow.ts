import type { ToastId } from '@invoke-ai/ui-library';
import { useToast } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import {
  formFieldInitialValuesChanged,
  workflowCategoryChanged,
  workflowIDChanged,
  workflowNameChanged,
} from 'features/nodes/store/nodesSlice';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { useGetFormFieldInitialValues } from 'features/workflowLibrary/hooks/useGetFormInitialValues';
import { newWorkflowSaved } from 'features/workflowLibrary/store/actions';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useCreateWorkflowMutation, useUpdateOpenedAtMutation, workflowsApi } from 'services/api/endpoints/workflows';
import type { SetFieldType } from 'type-fest';

/**
 * A draft workflow is a workflow that is has not been saved yet. It does not have an id and is not in the default category.
 */
type DraftWorkflow = SetFieldType<
  SetFieldType<WorkflowV3, 'id', undefined>,
  'meta',
  SetFieldType<WorkflowV3['meta'], 'category', Exclude<WorkflowV3['meta']['category'], 'default'>>
>;

export const isDraftWorkflow = (workflow: WorkflowV3): workflow is DraftWorkflow =>
  !workflow.id && workflow.meta.category !== 'default';

type CreateLibraryWorkflowArg = {
  workflow: DraftWorkflow;
  onSuccess?: () => void;
  onError?: () => void;
};

type CreateLibraryWorkflowReturn = {
  createNewWorkflow: (arg: CreateLibraryWorkflowArg) => Promise<void>;
  isLoading: boolean;
  isError: boolean;
};

export const useCreateLibraryWorkflow = (): CreateLibraryWorkflowReturn => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [createWorkflow, { isLoading, isError }] = useCreateWorkflowMutation();
  const [updateOpenedAt] = useUpdateOpenedAtMutation();
  const getFormFieldInitialValues = useGetFormFieldInitialValues();

  const toast = useToast();
  const toastRef = useRef<ToastId | undefined>();
  const createNewWorkflow = useCallback(
    async ({ workflow, onSuccess, onError }: CreateLibraryWorkflowArg) => {
      toastRef.current = toast({
        title: t('workflows.savingWorkflow'),
        status: 'loading',
        duration: null,
        isClosable: false,
      });
      try {
        const data = await createWorkflow(workflow).unwrap();
        const {
          id,
          name,
          meta: { category },
        } = data.workflow;
        dispatch(workflowIDChanged(id));
        dispatch(workflowNameChanged(name));
        dispatch(workflowCategoryChanged(category));
        dispatch(newWorkflowSaved({ category }));
        // When a workflow is saved, the form field initial values are updated to the current form field values
        dispatch(formFieldInitialValuesChanged({ formFieldInitialValues: getFormFieldInitialValues() }));
        updateOpenedAt({ workflow_id: id });
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
    [toast, t, createWorkflow, dispatch, getFormFieldInitialValues, updateOpenedAt]
  );
  return {
    createNewWorkflow,
    isLoading,
    isError,
  };
};
