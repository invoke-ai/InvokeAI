import { useToast } from '@invoke-ai/ui-library';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { useValidateAndLoadWorkflow } from 'features/workflowLibrary/hooks/useValidateAndLoadWorkflow';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyGetWorkflowQuery, useUpdateOpenedAtMutation, workflowsApi } from 'services/api/endpoints/workflows';

export const useLoadWorkflowFromLibrary = () => {
  const toast = useToast();
  const { t } = useTranslation();
  const validateAndLoadWorkflow = useValidateAndLoadWorkflow();
  const [getWorkflow] = useLazyGetWorkflowQuery();
  const [updateOpenedAt] = useUpdateOpenedAtMutation();
  const loadWorkflowFromLibrary = useCallback(
    async (
      workflowId: string,
      options: {
        onSuccess?: (workflow: WorkflowV3) => void;
        onError?: () => void;
      } = {}
    ) => {
      const { onSuccess, onError } = options;
      try {
        const res = await getWorkflow(workflowId).unwrap();

        const validatedWorkflow = await validateAndLoadWorkflow(res.workflow);

        if (!validatedWorkflow) {
          onError?.();
          return;
        }
        updateOpenedAt({ workflow_id: workflowId });
        onSuccess?.(validatedWorkflow);
      } catch {
        // This is catching the error from the getWorkflow query
        toast({
          id: `AUTH_ERROR_TOAST_${workflowsApi.endpoints.getWorkflow.name}`,
          title: t('toast.problemRetrievingWorkflow'),
          status: 'error',
        });
        onError?.();
      }
    },
    [getWorkflow, validateAndLoadWorkflow, updateOpenedAt, toast, t]
  );

  return loadWorkflowFromLibrary;
};
