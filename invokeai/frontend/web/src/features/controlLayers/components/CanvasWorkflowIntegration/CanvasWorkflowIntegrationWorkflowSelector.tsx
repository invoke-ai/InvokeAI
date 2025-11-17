import { Flex, FormControl, FormLabel, Select, Spinner, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  canvasWorkflowIntegrationWorkflowSelected,
  selectCanvasWorkflowIntegrationSelectedWorkflowId,
} from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListWorkflowsInfiniteInfiniteQuery } from 'services/api/endpoints/workflows';

export const CanvasWorkflowIntegrationWorkflowSelector = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const selectedWorkflowId = useAppSelector(selectCanvasWorkflowIntegrationSelectedWorkflowId);
  const { data: workflowsData, isLoading } = useListWorkflowsInfiniteInfiniteQuery(
    {
      per_page: 100, // Get a reasonable number of workflows
      page: 0,
    },
    {
      selectFromResult: ({ data, isLoading }) => ({
        data,
        isLoading,
      }),
    }
  );

  const workflows = useMemo(() => {
    if (!workflowsData) {
      return [];
    }
    // Flatten all pages into a single list
    return workflowsData.pages.flatMap((page) => page.items);
  }, [workflowsData]);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const workflowId = e.target.value || null;
      dispatch(canvasWorkflowIntegrationWorkflowSelected({ workflowId }));
    },
    [dispatch]
  );

  if (isLoading) {
    return (
      <Flex alignItems="center" gap={2}>
        <Spinner size="sm" />
        <Text>{t('controlLayers.workflowIntegration.loadingWorkflows', 'Loading workflows...')}</Text>
      </Flex>
    );
  }

  if (workflows.length === 0) {
    return (
      <Text color="warning.400" fontSize="sm">
        {t('controlLayers.workflowIntegration.noWorkflowsFound', 'No workflows found.')}
      </Text>
    );
  }

  return (
    <FormControl>
      <FormLabel>{t('controlLayers.workflowIntegration.selectWorkflow', 'Select Workflow')}</FormLabel>
      <Select
        placeholder={t('controlLayers.workflowIntegration.selectPlaceholder', 'Choose a workflow...')}
        value={selectedWorkflowId || ''}
        onChange={onChange}
      >
        {workflows.map((workflow) => (
          <option key={workflow.workflow_id} value={workflow.workflow_id}>
            {workflow.name || t('controlLayers.workflowIntegration.unnamedWorkflow', 'Unnamed Workflow')}
          </option>
        ))}
      </Select>
    </FormControl>
  );
});

CanvasWorkflowIntegrationWorkflowSelector.displayName = 'CanvasWorkflowIntegrationWorkflowSelector';
