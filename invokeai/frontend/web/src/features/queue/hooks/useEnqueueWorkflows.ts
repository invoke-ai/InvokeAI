import { createAction } from '@reduxjs/toolkit';
import { useAppStore } from 'app/store/nanostores/store';
import {
  $outputNodeId,
  getPublishInputs,
  selectFieldIdentifiersWithInvocationTypes,
} from 'features/nodes/components/sidePanel/workflow/publish';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodeData, selectNodesSlice } from 'features/nodes/store/selectors';
import { isBatchNode, isInvocationNode } from 'features/nodes/types/invocation';
import { buildNodesGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { resolveBatchValue } from 'features/nodes/util/node/resolveBatchValue';
import { buildWorkflowWithValidation } from 'features/nodes/util/workflow/buildWorkflow';
import { groupBy } from 'lodash-es';
import { useCallback } from 'react';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import type { Batch, EnqueueBatchArg } from 'services/api/types';
import { assert } from 'tsafe';

const enqueueRequestedWorkflows = createAction('app/enqueueRequestedWorkflows');

export const useEnqueueWorkflows = () => {
  const { getState, dispatch } = useAppStore();
  const enqueue = useCallback(
    async (prepend: boolean, isApiValidationRun: boolean) => {
      dispatch(enqueueRequestedWorkflows());
      const state = getState();
      const nodesState = selectNodesSlice(state);
      const templates = $templates.get();
      const graph = buildNodesGraph(state, templates);
      const builtWorkflow = buildWorkflowWithValidation(nodesState);

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
        // Derive the input fields from the builder's selected node field elements
        const fieldIdentifiers = selectFieldIdentifiersWithInvocationTypes(state);
        const inputs = getPublishInputs(fieldIdentifiers, templates);
        const api_input_fields = inputs.publishable.map(({ nodeId, fieldName }) => {
          return {
            kind: 'input',
            node_id: nodeId,
            field_name: fieldName,
          } as const;
        });

        // Derive the output fields from the builder's selected output node
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

        assert(nodesState.id, 'Workflow without ID cannot be used for API validation run');

        batchConfig.validation_run_data = {
          workflow_id: nodesState.id,
          input_fields: api_input_fields,
          output_fields: api_output_fields,
        };

        // If the batch is an API validation run, we only want to run it once
        batchConfig.batch.runs = 1;
      }

      const req = dispatch(
        queueApi.endpoints.enqueueBatch.initiate(batchConfig, { ...enqueueMutationFixedCacheKeyOptions, track: false })
      );

      const enqueueResult = await req.unwrap();
      return { batchConfig, enqueueResult };
    },
    [dispatch, getState]
  );

  return enqueue;
};
