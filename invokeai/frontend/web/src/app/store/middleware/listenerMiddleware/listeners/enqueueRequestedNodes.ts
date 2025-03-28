import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { parseify } from 'common/util/serialize';
import { $outputNodeId } from 'features/nodes/components/sidePanel/builder/deploy';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodeData, selectNodesSlice } from 'features/nodes/store/selectors';
import { selectNodeFieldElementsDeduped } from 'features/nodes/store/workflowSlice';
import { isBatchNode, isInvocationNode } from 'features/nodes/types/invocation';
import { buildNodesGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { resolveBatchValue } from 'features/nodes/util/node/resolveBatchValue';
import { buildWorkflowWithValidation } from 'features/nodes/util/workflow/buildWorkflow';
import { groupBy } from 'lodash-es';
import { serializeError } from 'serialize-error';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import type { Batch, EnqueueBatchArg } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('generation');

export const enqueueRequestedWorkflows = createAction<{ prepend: boolean; isApiValidationRun: boolean }>(
  'app/enqueueRequestedWorkflows'
);

export const addEnqueueRequestedNodes = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: enqueueRequestedWorkflows,
    effect: async (action, { getState, dispatch }) => {
      const { prepend, isApiValidationRun } = action.payload;
      const state = getState();
      const nodesState = selectNodesSlice(state);
      const workflow = state.workflow;
      const templates = $templates.get();
      const graph = buildNodesGraph(state, templates);
      const builtWorkflow = buildWorkflowWithValidation({
        nodes: nodesState.nodes,
        edges: nodesState.edges,
        workflow,
      });

      if (builtWorkflow) {
        // embedded workflows don't have an id
        delete builtWorkflow.id;
      }

      const data: Batch['data'] = [];

      const invocationNodes = nodesState.nodes.filter(isInvocationNode);
      const batchNodes = invocationNodes.filter(isBatchNode);

      // Handle zipping batch nodes. First group the batch nodes by their batch_group_id
      const groupedBatchNodes = groupBy(batchNodes, (node) => node.data.inputs['batch_group_id']?.value);

      // Then, we will create a batch data collection item for each group
      for (const [batchGroupId, batchNodes] of Object.entries(groupedBatchNodes)) {
        const zippedBatchDataCollectionItems: NonNullable<Batch['data']>[number] = [];

        for (const node of batchNodes) {
          const value = await resolveBatchValue({ nodesState, node, dispatch });
          const sourceHandle = node.data.type === 'image_batch' ? 'image' : 'value';
          const edgesFromBatch = nodesState.edges.filter(
            (e) => e.source === node.id && e.sourceHandle === sourceHandle
          );
          if (batchGroupId !== 'None') {
            // If this batch node has a batch_group_id, we will zip the data collection items
            for (const edge of edgesFromBatch) {
              if (!edge.targetHandle) {
                break;
              }
              zippedBatchDataCollectionItems.push({
                node_path: edge.target,
                field_name: edge.targetHandle,
                items: value,
              });
            }
          } else {
            // Otherwise add the data collection items to root of the batch so they are not zipped
            const productBatchDataCollectionItems: NonNullable<Batch['data']>[number] = [];
            for (const edge of edgesFromBatch) {
              if (!edge.targetHandle) {
                break;
              }
              productBatchDataCollectionItems.push({
                node_path: edge.target,
                field_name: edge.targetHandle,
                items: value,
              });
            }
            if (productBatchDataCollectionItems.length > 0) {
              data.push(productBatchDataCollectionItems);
            }
          }
        }

        // Finally, if this batch data collection item has any items, add it to the data array
        if (batchGroupId !== 'None' && zippedBatchDataCollectionItems.length > 0) {
          data.push(zippedBatchDataCollectionItems);
        }
      }

      const batchConfig: EnqueueBatchArg = {
        batch: {
          graph,
          workflow: builtWorkflow,
          runs: state.params.iterations,
          origin: 'workflows',
          destination: 'gallery',
          data,
        },
        prepend,
      };

      if (isApiValidationRun) {
        const nodeFieldElements = selectNodeFieldElementsDeduped(state);
        const outputNodeId = $outputNodeId.get();
        assert(outputNodeId !== null, 'Output node not selected');
        const outputNodeType = selectNodeData(selectNodesSlice(state), outputNodeId).type;
        const outputNodeTemplate = templates[outputNodeType];
        assert(outputNodeTemplate, `Template for node type ${outputNodeType} not found`);
        const outputFieldNames = Object.keys(outputNodeTemplate.outputs);
        const api_output_fields = outputFieldNames.map((fieldName) => {
          return {
            kind: 'output',
            node_id: outputNodeId,
            field_name: fieldName,
          } as const;
        });
        const api_input_fields = nodeFieldElements.map((el) => {
          const { nodeId, fieldName } = el.data.fieldIdentifier;
          return {
            kind: 'input',
            node_id: nodeId,
            field_name: fieldName,
          } as const;
        });
        batchConfig.is_api_validation_run = true;
        batchConfig.api_input_fields = api_input_fields;
        batchConfig.api_output_fields = api_output_fields;
      }

      const req = dispatch(queueApi.endpoints.enqueueBatch.initiate(batchConfig, enqueueMutationFixedCacheKeyOptions));
      try {
        await req.unwrap();
        log.debug(parseify({ batchConfig }), 'Enqueued batch');
      } catch (error) {
        log.error({ error: serializeError(error) }, 'Failed to enqueue batch');
      } finally {
        req.reset();
      }
    },
  });
};
