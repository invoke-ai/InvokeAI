import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { isBatchNode, isInvocationNode } from 'features/nodes/types/invocation';
import { buildNodesGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { buildWorkflowWithValidation } from 'features/nodes/util/workflow/buildWorkflow';
import { resolveBatchValue } from 'features/queue/store/readiness';
import { groupBy } from 'lodash-es';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import type { Batch, BatchConfig } from 'services/api/types';

export const addEnqueueRequestedNodes = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'workflows',
    effect: async (action, { getState, dispatch }) => {
      const state = getState();
      const nodes = selectNodesSlice(state);
      const workflow = state.workflow;
      const graph = buildNodesGraph(nodes);
      const builtWorkflow = buildWorkflowWithValidation({
        nodes: nodes.nodes,
        edges: nodes.edges,
        workflow,
      });

      if (builtWorkflow) {
        // embedded workflows don't have an id
        delete builtWorkflow.id;
      }

      const data: Batch['data'] = [];

      const invocationNodes = nodes.nodes.filter(isInvocationNode);
      const batchNodes = invocationNodes.filter(isBatchNode);

      // Handle zipping batch nodes. First group the batch nodes by their batch_group_id
      const groupedBatchNodes = groupBy(batchNodes, (node) => node.data.inputs['batch_group_id']?.value);

      // Then, we will create a batch data collection item for each group
      for (const [batchGroupId, batchNodes] of Object.entries(groupedBatchNodes)) {
        const zippedBatchDataCollectionItems: NonNullable<Batch['data']>[number] = [];

        for (const node of batchNodes) {
          const value = resolveBatchValue(node, invocationNodes, nodes.edges);
          const sourceHandle = node.data.type === 'image_batch' ? 'image' : 'value';
          const edgesFromBatch = nodes.edges.filter((e) => e.source === node.id && e.sourceHandle === sourceHandle);
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

      const batchConfig: BatchConfig = {
        batch: {
          graph,
          workflow: builtWorkflow,
          runs: state.params.iterations,
          origin: 'workflows',
          destination: 'gallery',
          data,
        },
        prepend: action.payload.prepend,
      };

      const req = dispatch(queueApi.endpoints.enqueueBatch.initiate(batchConfig, enqueueMutationFixedCacheKeyOptions));
      try {
        await req.unwrap();
      } finally {
        req.reset();
      }
    },
  });
};
