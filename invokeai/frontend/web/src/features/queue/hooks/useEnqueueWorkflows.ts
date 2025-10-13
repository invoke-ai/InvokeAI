import { createAction } from '@reduxjs/toolkit';
import type { AppDispatch, AppStore, RootState } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { groupBy } from 'es-toolkit/compat';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import { isBatchNode, isInvocationNode } from 'features/nodes/types/invocation';
import { buildNodesGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { resolveBatchValue } from 'features/nodes/util/node/resolveBatchValue';
import { buildWorkflowWithValidation } from 'features/nodes/util/workflow/buildWorkflow';
import { useCallback } from 'react';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import type { Batch, EnqueueBatchArg } from 'services/api/types';

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

const enqueueWorkflows = async (store: AppStore, templates: Templates, prepend: boolean) => {
  const { dispatch, getState } = store;

  dispatch(enqueueRequestedWorkflows());
  const state = getState();
  const nodesState = selectNodesSlice(state);
  const graph = buildNodesGraph(state, templates);
  const workflow = buildWorkflowWithValidation(nodesState);

  if (workflow) {
    // embedded workflows don't have an id
    delete workflow.id;
  }

  const runs = state.params.iterations;
  const data = await getBatchDataForWorkflowGeneration(state, dispatch);

  const batchConfig: EnqueueBatchArg = {
    batch: {
      graph,
      workflow,
      runs,
      origin: 'workflows',
      destination: 'gallery',
      data,
    },
    prepend,
  };

  const req = dispatch(
    queueApi.endpoints.enqueueBatch.initiate(batchConfig, { ...enqueueMutationFixedCacheKeyOptions, track: false })
  );

  const enqueueResult = await req.unwrap();
  return { batchConfig, enqueueResult };
};

export const useEnqueueWorkflows = () => {
  const store = useAppStore();
  const enqueue = useCallback(
    (prepend: boolean) => {
      return enqueueWorkflows(store, $templates.get(), prepend);
    },
    [store]
  );

  return enqueue;
};
