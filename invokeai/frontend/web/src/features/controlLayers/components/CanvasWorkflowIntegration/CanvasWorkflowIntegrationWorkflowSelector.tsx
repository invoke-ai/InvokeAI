import { Flex, FormControl, FormLabel, Select, Spinner, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  canvasWorkflowIntegrationWorkflowSelected,
  selectCanvasWorkflowIntegrationSelectedWorkflowId,
} from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListWorkflowsInfiniteInfiniteQuery } from 'services/api/endpoints/workflows';

export const CanvasWorkflowIntegrationWorkflowSelector = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const selectedWorkflowId = useAppSelector(selectCanvasWorkflowIntegrationSelectedWorkflowId);
  const { data: workflowsData, isLoading } = useListWorkflowsInfiniteInfiniteQuery(
    {
      queryArg: {
        per_page: 100, // Get a reasonable number of workflows
        page: 0,
      },
      pageParam: 0,
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
      return null;
    }
    // Flatten all pages into a single list
    return {
      items: workflowsData.pages.flatMap((page) => page.items),
    };
  }, [workflowsData]);

  // Filter workflows that have image input parameters
  const compatibleWorkflows = useMemo(() => {
    if (!workflows) {
      return [];
    }

    return workflows.items.filter((workflow) => {
      // Check if the workflow has exposed fields
      if (!workflow.exposedFields || workflow.exposedFields.length === 0) {
        return false;
      }

      // Check if any of the nodes have image input fields
      const hasImageInput = workflow.nodes.some((node) => {
        return Object.values(node.data.inputs || {}).some((input) => {
          // @ts-expect-error - input may not have type property
          return input.type?.name === 'ImageField';
        });
      });

      return hasImageInput;
    });
  }, [workflows]);

  const onChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
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

  if (compatibleWorkflows.length === 0) {
    return (
      <Text color="warning.400" fontSize="sm">
        {t(
          'controlLayers.workflowIntegration.noCompatibleWorkflows',
          'No compatible workflows found. Workflows must have image input parameters and exposed fields configured in linear view.'
        )}
      </Text>
    );
  }

  return (
    <FormControl>
      <FormLabel>{t('controlLayers.workflowIntegration.selectWorkflow', 'Select Workflow')}</FormLabel>
      <Select placeholder={t('controlLayers.workflowIntegration.selectPlaceholder', 'Choose a workflow...')} value={selectedWorkflowId || ''} onChange={onChange}>
        {compatibleWorkflows.map((workflow) => (
          <option key={workflow.workflow_id} value={workflow.workflow_id}>
            {workflow.name || t('controlLayers.workflowIntegration.unnamedWorkflow', 'Unnamed Workflow')}
          </option>
        ))}
      </Select>
    </FormControl>
  );
});

CanvasWorkflowIntegrationWorkflowSelector.displayName = 'CanvasWorkflowIntegrationWorkflowSelector';
