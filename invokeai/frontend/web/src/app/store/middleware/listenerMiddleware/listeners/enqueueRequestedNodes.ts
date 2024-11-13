import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { isImageBatchNode } from 'features/nodes/types/invocation';
import { buildNodesGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { buildWorkflowWithValidation } from 'features/nodes/util/workflow/buildWorkflow';
import { queueApi } from 'services/api/endpoints/queue';
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

      // Skip edges from batch nodes - these should not be in the graph, they exist only in the UI
      const imageBatchNodes = nodes.nodes.filter(isImageBatchNode);
      for (const imageBatch of imageBatchNodes) {
        const edge = nodes.edges.find((e) => e.source === imageBatch.id);
        if (!edge || !edge.targetHandle) {
          break;
        }
        data.push([{ node_path: edge.target, field_name: edge.targetHandle, items: imageBatch.data.images }]);
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

      const req = dispatch(
        queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
          fixedCacheKey: 'enqueueBatch',
        })
      );
      try {
        await req.unwrap();
      } finally {
        req.reset();
      }
    },
  });
};
