import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectCanvasSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';
import {
  canvasWorkflowIntegrationClosed,
  canvasWorkflowIntegrationProcessingCompleted,
  canvasWorkflowIntegrationProcessingStarted,
  selectCanvasWorkflowIntegrationFieldValues,
  selectCanvasWorkflowIntegrationSelectedImageFieldKey,
  selectCanvasWorkflowIntegrationSelectedWorkflowId,
  selectCanvasWorkflowIntegrationSourceEntityIdentifier,
} from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import { CANVAS_OUTPUT_PREFIX, getPrefixedId } from 'features/nodes/util/graph/graphBuilderUtils';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { queueApi } from 'services/api/endpoints/queue';
import { useLazyGetWorkflowQuery } from 'services/api/endpoints/workflows';

const log = logger('canvas-workflow-integration');

export const useCanvasWorkflowIntegrationExecute = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();

  const selectedWorkflowId = useAppSelector(selectCanvasWorkflowIntegrationSelectedWorkflowId);
  const sourceEntityIdentifier = useAppSelector(selectCanvasWorkflowIntegrationSourceEntityIdentifier);
  const fieldValues = useAppSelector(selectCanvasWorkflowIntegrationFieldValues);
  const selectedImageFieldKey = useAppSelector(selectCanvasWorkflowIntegrationSelectedImageFieldKey);
  const canvasSessionId = useAppSelector(selectCanvasSessionId);

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
      // Use the user-selected ImageField, or find one automatically
      let imageFieldIdentifier: { nodeId: string; fieldName: string } | undefined;

      // Method 1: Use user-selected ImageField (preferred)
      if (selectedImageFieldKey) {
        const [nodeId, fieldName] = selectedImageFieldKey.split('.');
        if (nodeId && fieldName) {
          imageFieldIdentifier = { nodeId, fieldName };
        }
      }

      // Method 2: Search through form elements for an ImageField (fallback)
      if (!imageFieldIdentifier && workflow.workflow.form && workflow.workflow.form.elements) {
        for (const element of Object.values(workflow.workflow.form.elements)) {
          if (element.type !== 'node-field') {
            continue;
          }

          const fieldIdentifier = element.data?.fieldIdentifier;
          if (!fieldIdentifier) {
            continue;
          }

          // @ts-expect-error - node data type is complex
          const node = workflow.workflow.nodes.find((n) => n.data.id === fieldIdentifier.nodeId);
          if (!node) {
            continue;
          }

          // @ts-expect-error - node.data type is complex
          if (node.data.type === 'image') {
            imageFieldIdentifier = fieldIdentifier;
            break;
          }

          // Check if field type is ImageField
          // @ts-expect-error - field type is complex
          const field = node.data.inputs[fieldIdentifier.fieldName];
          if (field?.type?.name === 'ImageField') {
            imageFieldIdentifier = fieldIdentifier;
            break;
          }
        }
      }

      // Method 3: Fallback to exposedFields
      if (!imageFieldIdentifier && workflow.workflow.exposedFields) {
        imageFieldIdentifier = workflow.workflow.exposedFields.find((fieldIdentifier) => {
          // @ts-expect-error - node data type is complex
          const node = workflow.workflow.nodes.find((n) => n.data.id === fieldIdentifier.nodeId);
          if (!node) {
            return false;
          }

          // @ts-expect-error - node.data type is complex
          if (node.data.type === 'image') {
            return true;
          }

          // Check if field type is ImageField
          // @ts-expect-error - field type is complex
          const field = node.data.inputs[fieldIdentifier.fieldName];
          return field?.type?.name === 'ImageField';
        });
      }

      if (!imageFieldIdentifier) {
        throw new Error('Workflow does not have an image input field in the Form Builder');
      }

      // Update the workflow nodes with canvas image and user field values
      const updatedWorkflow = {
        ...workflow.workflow,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodes: workflow.workflow.nodes.map((node: any) => {
          const nodeId = node.data.id;
          let updatedInputs = { ...node.data.inputs };
          const updatedData = { ...node.data };
          let hasChanges = false;

          // Apply image field if this is the image node
          if (nodeId === imageFieldIdentifier.nodeId) {
            updatedInputs[imageFieldIdentifier.fieldName] = {
              ...updatedInputs[imageFieldIdentifier.fieldName],
              value: imageDTO,
            };
            hasChanges = true;
          }

          // Apply any field values from Redux state
          if (fieldValues) {
            Object.entries(fieldValues).forEach(([fieldKey, value]) => {
              const [fieldNodeId, fieldName] = fieldKey.split('.');
              if (fieldNodeId && fieldName && fieldNodeId === nodeId && updatedInputs[fieldName]) {
                updatedInputs[fieldName] = {
                  ...updatedInputs[fieldName],
                  value: value,
                };
                hasChanges = true;
              }
            });
          }

          // If anything was modified, return updated node
          if (hasChanges) {
            updatedData.inputs = updatedInputs;
            return {
              ...node,
              data: updatedData,
            };
          }

          return node;
        }),
      };

      // Validate that the workflow has a canvas_output node
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const hasCanvasOutputNode = updatedWorkflow.nodes.some((node: any) => node.data?.type === 'canvas_output');
      if (!hasCanvasOutputNode) {
        throw new Error('Workflow does not have a Canvas Output node');
      }

      // 4. Convert workflow to graph format
      const graphNodes: Record<string, unknown> = {};
      const nodeIdMapping: Record<string, string> = {}; // Map original IDs to new IDs

      for (const node of updatedWorkflow.nodes) {
        const nodeData = node.data;
        const isCanvasOutputNode = nodeData.type === 'canvas_output';

        // Prefix canvas_output node IDs so the staging area can find them
        const nodeId = isCanvasOutputNode ? getPrefixedId(CANVAS_OUTPUT_PREFIX) : nodeData.id;
        nodeIdMapping[nodeData.id] = nodeId;

        const invocation: Record<string, unknown> = {
          id: nodeId,
          type: nodeData.type,
          // Canvas output nodes are always intermediate (they go to the staging area, not gallery)
          is_intermediate: isCanvasOutputNode ? true : (nodeData.isIntermediate ?? false),
          use_cache: nodeData.useCache ?? true,
        };

        // Add input values to the invocation
        for (const [fieldName, fieldData] of Object.entries(nodeData.inputs)) {
          const fieldValue = (fieldData as { value?: unknown }).value;
          if (fieldValue === undefined) {
            continue;
          }

          // The frontend stores board fields as 'auto', 'none', or { board_id: string }.
          // The backend expects null or { board_id: string }. Translate accordingly.
          if (fieldName === 'board') {
            if (fieldValue === 'auto' || fieldValue === 'none' || fieldValue === null) {
              continue;
            }
          }

          invocation[fieldName] = fieldValue;
        }

        graphNodes[nodeId] = invocation;
      }

      // Convert edges to graph format, using the node ID mapping
      const edgesArray = updatedWorkflow.edges as Array<{
        source: string;
        target: string;
        sourceHandle: string;
        targetHandle: string;
      }>;
      const graphEdges = edgesArray.map((edge) => ({
        source: {
          node_id: nodeIdMapping[edge.source] || edge.source,
          field: edge.sourceHandle,
        },
        destination: {
          node_id: nodeIdMapping[edge.target] || edge.target,
          field: edge.targetHandle,
        },
      }));

      const graph = {
        id: workflow.workflow.id || workflow.workflow_id || 'temp',
        nodes: graphNodes,
        edges: graphEdges,
      };

      log.debug({ workflowName: workflow.name, destination: canvasSessionId }, 'Enqueueing workflow on canvas');

      await dispatch(
        queueApi.endpoints.enqueueBatch.initiate({
          batch: {
            workflow: updatedWorkflow,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            graph: graph as any,
            runs: 1,
            origin: 'canvas_workflow_integration',
            destination: canvasSessionId,
          },
          prepend: true,
        })
      ).unwrap();

      // 5. Close the modal and show success message
      // Results will appear in the staging area where user can accept/discard them
      toast({
        status: 'success',
        title: t('controlLayers.workflowIntegration.executionStarted', 'Workflow execution started'),
        description: t(
          'controlLayers.workflowIntegration.executionStartedDescription',
          'The result will appear in the staging area when complete.'
        ),
      });

      dispatch(canvasWorkflowIntegrationClosed());
    } catch (error) {
      log.error('Error executing workflow');
      dispatch(canvasWorkflowIntegrationProcessingCompleted());
      toast({
        status: 'error',
        title: t('controlLayers.workflowIntegration.executionFailed', 'Failed to execute workflow'),
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }, [
    selectedWorkflowId,
    sourceEntityIdentifier,
    canvasManager,
    dispatch,
    getWorkflow,
    t,
    fieldValues,
    selectedImageFieldKey,
    canvasSessionId,
  ]);

  return {
    execute,
    canExecute,
  };
};
