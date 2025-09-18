import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectCanvasSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { $templates } from 'features/nodes/store/nodesSlice';
import { buildInvocationGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { CANVAS_OUTPUT_PREFIX } from 'features/nodes/util/graph/graphBuilderUtils';
import { toast } from 'features/toast/toast';
import { setWorkflowLibraryBrowseIntent } from 'features/workflowLibrary/store/workflowLibraryIntent';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import type { EnqueueBatchArg } from 'services/api/types';

export const useWorkflowTriggerApply = () => {
  const canvasManager = useCanvasManager();
  const workflowState = useStore(canvasManager.stateApi.$workflowTrigger);
  const selection = workflowState?.selection ?? null;
  const selectedWorkflow = selection?.workflow ?? null;
  const selectedWorkflowName = selection?.workflowName ?? null;
  const canvasSessionId = useAppSelector(selectCanvasSessionId);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [isApplying, setIsApplying] = useState(false);
  const templates = useStore($templates);

  const apply = useCallback(async () => {
    if (!selection || !selectedWorkflow || !canvasSessionId) {
      return;
    }

    setIsApplying(true);
    try {
      const workflow = deepClone(selectedWorkflow);
      const outputFields = workflow.output_fields ?? [];
      if (outputFields.length === 0) {
        toast({
          status: 'error',
          title: t('controlLayers.triggerWorkflow.noWorkflowSelected'),
        });
        return;
      }

      const outputField = outputFields[0]!;
      const originalNodeId = outputField.nodeId;
      const fieldName = outputField.fieldName ?? outputField.field_name;
      if (!originalNodeId || !fieldName) {
        toast({
          status: 'error',
          title: t('controlLayers.triggerWorkflow.noWorkflowSelected'),
        });
        return;
      }
      const outputNodeId = `${CANVAS_OUTPUT_PREFIX}:${selection.workflowId}`;

      workflow.output_fields = [
        {
          ...outputField,
          kind: 'output',
          nodeId: outputNodeId,
          node_id: outputNodeId,
          fieldName,
          field_name: fieldName,
          userLabel: outputField.userLabel ?? outputField.user_label ?? null,
          user_label: outputField.userLabel ?? outputField.user_label ?? null,
        },
      ];

      workflow.nodes = workflow.nodes.map((node) => {
        if (node.id !== originalNodeId || node.type !== 'invocation') {
          return node;
        }
        return {
          ...node,
          id: outputNodeId,
          data: { ...node.data, id: outputNodeId },
        };
      });

      workflow.edges = workflow.edges.map((edge) => {
        if (edge.type !== 'default') {
          return edge;
        }
        return {
          ...edge,
          source: edge.source === originalNodeId ? outputNodeId : edge.source,
          target: edge.target === originalNodeId ? outputNodeId : edge.target,
        };
      });

      const graph = buildInvocationGraph({
        nodes: workflow.nodes,
        edges: workflow.edges,
        templates,
        graphId: workflow.id,
      });

      const workflowPayload = deepClone(workflow);
      delete workflowPayload.id;

      const enqueueArg: EnqueueBatchArg = {
        batch: {
          graph,
          workflow: workflowPayload,
          runs: 1,
          origin: 'canvas',
          destination: canvasSessionId,
        },
      };

      const req = dispatch(
        queueApi.endpoints.enqueueBatch.initiate(enqueueArg, {
          ...enqueueMutationFixedCacheKeyOptions,
          track: false,
        })
      );

      await req.unwrap();

      toast({
        status: 'success',
        title: t('controlLayers.triggerWorkflow.enqueued'),
      });

      setWorkflowLibraryBrowseIntent();
      canvasManager.stateApi.cancelWorkflowTrigger();
    } catch {
      toast({
        status: 'error',
        title: t('controlLayers.triggerWorkflow.enqueueFailed'),
      });
    } finally {
      setIsApplying(false);
    }
  }, [canvasManager.stateApi, canvasSessionId, dispatch, selection, selectedWorkflow, t, templates]);

  return useMemo(
    () => ({ apply, isApplying, selection, selectedWorkflow, selectedWorkflowName }) as const,
    [apply, isApplying, selection, selectedWorkflow, selectedWorkflowName]
  );
};
