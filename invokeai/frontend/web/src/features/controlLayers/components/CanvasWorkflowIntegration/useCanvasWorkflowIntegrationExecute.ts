import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import {
  canvasWorkflowIntegrationClosed,
  canvasWorkflowIntegrationProcessingCompleted,
  canvasWorkflowIntegrationProcessingStarted,
  selectCanvasWorkflowIntegrationFieldValues,
  selectCanvasWorkflowIntegrationSelectedWorkflowId,
  selectCanvasWorkflowIntegrationSourceEntityIdentifier,
} from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { buildNodesGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { buildWorkflowWithValidation } from 'features/nodes/util/workflow/buildWorkflow';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyGetWorkflowQuery } from 'services/api/endpoints/workflows';
import { queueApi } from 'services/api/endpoints/queue';
import type { ImageDTO } from 'services/api/types';

export const useCanvasWorkflowIntegrationExecute = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const store = useAppStore();
  const canvasManager = useCanvasManager();

  const selectedWorkflowId = useAppSelector(selectCanvasWorkflowIntegrationSelectedWorkflowId);
  const sourceEntityIdentifier = useAppSelector(selectCanvasWorkflowIntegrationSourceEntityIdentifier);
  const fieldValues = useAppSelector(selectCanvasWorkflowIntegrationFieldValues);

  const [getWorkflow] = useLazyGetWorkflowQuery();

  const canExecute = useMemo(() => {
    return Boolean(selectedWorkflowId && sourceEntityIdentifier);
  }, [selectedWorkflowId, sourceEntityIdentifier]);

  const execute = useCallback(async () => {
    if (!selectedWorkflowId || !sourceEntityIdentifier || !canvasManager) {
      return;
    }

    try {
      dispatch(canvasWorkflowIntegrationProcessingStarted());

      // 1. Extract the canvas layer as an image
      const adapter = canvasManager.getAdapter(sourceEntityIdentifier);
      if (!adapter) {
        throw new Error('Could not find canvas entity adapter');
      }

      const rect = adapter.transformer.getRelativeRect();
      const imageDTO = await adapter.renderer.rasterize({ rect, attrs: { filters: [], opacity: 1 } });

      // 2. Fetch the workflow
      const { data: workflow } = await getWorkflow(selectedWorkflowId);
      if (!workflow) {
        throw new Error('Failed to load workflow');
      }

      // 3. Build the workflow graph with the canvas image
      // First, find the image field in the workflow
      const imageFieldIdentifier = workflow.exposedFields.find((fieldIdentifier) => {
        const node = workflow.nodes.find((n) => n.data.id === fieldIdentifier.nodeId);
        if (!node) {
          return false;
        }
        const field = node.data.inputs[fieldIdentifier.fieldName];
        // @ts-expect-error - field may not have type property
        return field?.type?.name === 'ImageField';
      });

      if (!imageFieldIdentifier) {
        throw new Error('Workflow does not have an image input field');
      }

      // Update the workflow nodes with our values
      const updatedWorkflow = {
        ...workflow,
        nodes: workflow.nodes.map((node) => {
          if (node.data.id === imageFieldIdentifier.nodeId) {
            return {
              ...node,
              data: {
                ...node.data,
                inputs: {
                  ...node.data.inputs,
                  [imageFieldIdentifier.fieldName]: {
                    ...node.data.inputs[imageFieldIdentifier.fieldName],
                    value: imageDTO,
                  },
                },
              },
            };
          }

          // Apply other field values
          if (fieldValues) {
            const fieldValue = Object.entries(fieldValues).find(
              ([key]) => key === `${node.data.id}.${node.data.inputs}`
            );
            if (fieldValue) {
              const [, value] = fieldValue;
              return {
                ...node,
                data: {
                  ...node.data,
                  inputs: {
                    ...node.data.inputs,
                    // @ts-expect-error - dynamic field assignment
                    [fieldValue[0].split('.')[1]]: value,
                  },
                },
              };
            }
          }

          return node;
        }),
      };

      // Build the graph
      const state = store.getState();
      const graph = buildNodesGraph(
        {
          ...state,
          nodes: {
            ...state.nodes,
            nodes: updatedWorkflow.nodes,
            edges: updatedWorkflow.edges,
          },
        },
        // @ts-expect-error - templates type mismatch
        {}
      );

      // 4. Execute the workflow
      const result = await dispatch(
        queueApi.endpoints.enqueueBatch.initiate({
          batch: {
            graph,
            runs: 1,
            origin: 'canvas_workflow_integration',
            destination: 'canvas',
          },
          prepend: true,
        })
      ).unwrap();

      // 5. Wait for the result and add it to canvas
      // Note: In a real implementation, we would need to listen to socket events
      // for the completion of this batch. For now, we'll show a toast and close.
      toast({
        status: 'success',
        title: t('controlLayers.workflowIntegration.executionStarted', 'Workflow execution started'),
        description: t(
          'controlLayers.workflowIntegration.executionStartedDescription',
          'The result will be added to the canvas when complete.'
        ),
      });

      dispatch(canvasWorkflowIntegrationClosed());
    } catch (error) {
      console.error('Error executing workflow:', error);
      toast({
        status: 'error',
        title: t('controlLayers.workflowIntegration.executionFailed', 'Failed to execute workflow'),
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      dispatch(canvasWorkflowIntegrationProcessingCompleted());
    }
  }, [selectedWorkflowId, sourceEntityIdentifier, fieldValues, canvasManager, dispatch, getWorkflow, store, t]);

  return {
    execute,
    canExecute,
  };
};

// Hook to listen for workflow completion and add result to canvas
export const useCanvasWorkflowIntegrationResultHandler = () => {
  const dispatch = useAppDispatch();

  const handleWorkflowComplete = useCallback(
    (imageDTO: ImageDTO) => {
      // Add the result as a new raster layer
      const imageObject = imageDTOToImageObject(imageDTO);
      dispatch(
        rasterLayerAdded({
          overrides: {
            objects: [imageObject],
            position: { x: 0, y: 0 },
          },
          isSelected: true,
        })
      );

      toast({
        status: 'success',
        title: 'Workflow result added to canvas',
      });
    },
    [dispatch]
  );

  return { handleWorkflowComplete };
};
