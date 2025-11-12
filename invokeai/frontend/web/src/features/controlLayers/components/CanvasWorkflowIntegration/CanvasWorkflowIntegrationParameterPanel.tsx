import { Box, Flex, Heading, Spinner, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { canvasWorkflowIntegrationFieldValueChanged, selectCanvasWorkflowIntegrationSelectedWorkflowId } from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import { InputFieldRenderer } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldRenderer';
import { useWorkflowFieldInstances } from 'features/nodes/hooks/useWorkflowFieldInstances';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetWorkflowQuery } from 'services/api/endpoints/workflows';

export const CanvasWorkflowIntegrationParameterPanel = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const selectedWorkflowId = useAppSelector(selectCanvasWorkflowIntegrationSelectedWorkflowId);

  const { data: workflow, isLoading } = useGetWorkflowQuery(selectedWorkflowId!, {
    skip: !selectedWorkflowId,
  });

  const onFieldValueChanged = useCallback(
    (fieldName: string, value: unknown) => {
      dispatch(canvasWorkflowIntegrationFieldValueChanged({ fieldName, value }));
    },
    [dispatch]
  );

  if (isLoading) {
    return (
      <Flex alignItems="center" gap={2}>
        <Spinner size="sm" />
        <Text>{t('controlLayers.workflowIntegration.loadingParameters', 'Loading workflow parameters...')}</Text>
      </Flex>
    );
  }

  if (!workflow) {
    return null;
  }

  // Get exposed fields that are NOT image fields (those will be auto-populated)
  const exposedFieldsToShow = workflow.exposedFields.filter((fieldIdentifier) => {
    const node = workflow.nodes.find((n) => n.data.id === fieldIdentifier.nodeId);
    if (!node) {
      return false;
    }

    const field = node.data.inputs[fieldIdentifier.fieldName];
    // @ts-expect-error - field may not have type property
    return field?.type?.name !== 'ImageField';
  });

  if (exposedFieldsToShow.length === 0) {
    return (
      <Box>
        <Text fontSize="sm" color="base.400">
          {t(
            'controlLayers.workflowIntegration.noParametersToCustomize',
            'This workflow has no customizable parameters (all image fields will be auto-populated).'
          )}
        </Text>
      </Box>
    );
  }

  return (
    <Box>
      <Heading size="sm" mb={3}>
        {t('controlLayers.workflowIntegration.parameters', 'Workflow Parameters')}
      </Heading>
      <Flex direction="column" gap={3}>
        {exposedFieldsToShow.map((fieldIdentifier) => {
          const node = workflow.nodes.find((n) => n.data.id === fieldIdentifier.nodeId);
          if (!node) {
            return null;
          }

          return (
            <Box key={`${fieldIdentifier.nodeId}.${fieldIdentifier.fieldName}`}>
              <InputFieldRenderer
                nodeId={fieldIdentifier.nodeId}
                fieldName={fieldIdentifier.fieldName}
                // @ts-expect-error - settings type mismatch
                settings={undefined}
              />
            </Box>
          );
        })}
      </Flex>
    </Box>
  );
});

CanvasWorkflowIntegrationParameterPanel.displayName = 'CanvasWorkflowIntegrationParameterPanel';
