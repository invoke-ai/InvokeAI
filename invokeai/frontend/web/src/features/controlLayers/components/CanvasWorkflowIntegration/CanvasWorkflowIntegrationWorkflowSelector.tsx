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

import { useFilteredWorkflows } from './useFilteredWorkflows';

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

  // Filter workflows to only show those with ImageFields
  const { filteredWorkflows, isFiltering } = useFilteredWorkflows(workflows);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const workflowId = e.target.value || null;
      dispatch(canvasWorkflowIntegrationWorkflowSelected({ workflowId }));
    },
    [dispatch]
  );

  if (isLoading || isFiltering) {
    return (
      <Flex alignItems="center" gap={2}>
        <Spinner size="sm" />
        <Text>
          {isFiltering
            ? t('controlLayers.workflowIntegration.filteringWorkflows')
            : t('controlLayers.workflowIntegration.loadingWorkflows')}
        </Text>
      </Flex>
    );
  }

  if (filteredWorkflows.length === 0) {
    return (
      <Text color="warning.400" fontSize="sm">
        {workflows.length === 0
          ? t('controlLayers.workflowIntegration.noWorkflowsFound')
          : t('controlLayers.workflowIntegration.noWorkflowsWithImageField')}
      </Text>
    );
  }

  return (
    <FormControl>
      <FormLabel>{t('controlLayers.workflowIntegration.selectWorkflow')}</FormLabel>
      <Select
        placeholder={t('controlLayers.workflowIntegration.selectPlaceholder')}
        value={selectedWorkflowId || ''}
        onChange={onChange}
      >
        {filteredWorkflows.map((workflow) => (
          <option key={workflow.workflow_id} value={workflow.workflow_id}>
            {workflow.name || t('controlLayers.workflowIntegration.unnamedWorkflow')}
          </option>
        ))}
      </Select>
    </FormControl>
  );
});

CanvasWorkflowIntegrationWorkflowSelector.displayName = 'CanvasWorkflowIntegrationWorkflowSelector';
