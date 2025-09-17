import { createAction } from '@reduxjs/toolkit';
import type { AppDispatch, RootState } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { groupBy } from 'es-toolkit/compat';
import {
  $outputNodeId,
  getPublishInputs,
  selectFieldIdentifiersWithInvocationTypes,
} from 'features/nodes/components/sidePanel/workflow/publish';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodeData, selectNodesSlice } from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import { isBatchNode, isInvocationNode } from 'features/nodes/types/invocation';
import { buildNodesGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { resolveBatchValue } from 'features/nodes/util/node/resolveBatchValue';
import { buildWorkflowWithValidation } from 'features/nodes/util/workflow/buildWorkflow';
import { useCallback } from 'react';
import type { Batch, EnqueueBatchArg, S } from 'services/api/types';
import { assert } from 'tsafe';

import { executeEnqueue } from './utils/executeEnqueue';

export const enqueueRequestedWorkflows = createAction('app/enqueueRequestedWorkflows');

const getBatchDataForWorkflowGeneration = async (state: RootState, dispatch: AppDispatch): Promise<Batch['data']> => {
  const nodesState = selectNodesSlice(state);
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
      const edgesFromBatch = nodesState.edges.filter((e) => e.source === node.id && e.sourceHandle === sourceHandle);
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

  return data;
};

const getValidationRunData = (state: RootState, templates: Templates): S['ValidationRunData'] => {
  const nodesState = selectNodesSlice(state);

  // Derive the input fields from the builder's selected node field elements
  const fieldIdentifiers = selectFieldIdentifiersWithInvocationTypes(state);
  const inputs = getPublishInputs(fieldIdentifiers, templates);
  const api_input_fields = inputs.publishable.map(({ nodeId, fieldName, label }) => {
    return {
      kind: 'input',
      node_id: nodeId,
      field_name: fieldName,
      user_label: label,
    } satisfies S['FieldIdentifier'];
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
      user_label: null,
    } satisfies S['FieldIdentifier'];
  });

  assert(nodesState.id, 'Workflow without ID cannot be used for API validation run');

  return {
    workflow_id: nodesState.id,
    input_fields: api_input_fields,
    output_fields: api_output_fields,
  };
};

export const useEnqueueWorkflows = () => {
  const store = useAppStore();
  const enqueue = useCallback(
    (prepend: boolean, isApiValidationRun: boolean) => {
      return executeEnqueue({
        store,
        options: { prepend, isApiValidationRun },
        requestedAction: enqueueRequestedWorkflows,
        build: async ({ store: innerStore, options }) => {
          const { dispatch, getState } = innerStore;
          const state = getState();
          const nodesState = selectNodesSlice(state);
          const templates = $templates.get();
          const graph = buildNodesGraph(state, templates);
          const workflow = buildWorkflowWithValidation(nodesState);

          if (workflow) {
            // embedded workflows don't have an id
            delete workflow.id;
          }

          const data = await getBatchDataForWorkflowGeneration(state, dispatch);

          const batchConfig: EnqueueBatchArg = {
            batch: {
              graph,
              workflow,
              runs: state.params.iterations,
              origin: 'workflows',
              destination: 'gallery',
              data,
            },
            prepend: options.prepend,
          };

          if (options.isApiValidationRun) {
            batchConfig.validation_run_data = getValidationRunData(state, templates);
            batchConfig.batch.runs = 1;
          }

          return { batchConfig } satisfies { batchConfig: EnqueueBatchArg };
        },
        prepareBatch: ({ buildResult }) => buildResult.batchConfig,
      });
    },
    [store]
  );

  return enqueue;
};
