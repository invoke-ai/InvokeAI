import { enqueueRequested } from 'app/store/actions';
import { buildNodesGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { buildWorkflow } from 'features/nodes/util/workflow/buildWorkflow';
import { queueApi } from 'services/api/endpoints/queue';
import type { BatchConfig } from 'services/api/types';

import { startAppListening } from '..';

export const addEnqueueRequestedNodes = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'nodes',
    effect: async (action, { getState, dispatch }) => {
      const state = getState();
      const { nodes, edges } = state.nodes;
      const workflow = state.workflow;
      const graph = buildNodesGraph(state.nodes);
      const builtWorkflow = buildWorkflow({
        nodes,
        edges,
        workflow,
      });

      // embedded workflows don't have an id
      delete builtWorkflow.id;

      const batchConfig: BatchConfig = {
        batch: {
          graph,
          workflow: builtWorkflow,
          runs: state.generation.iterations,
        },
        prepend: action.payload.prepend,
      };

      const req = dispatch(
        queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
          fixedCacheKey: 'enqueueBatch',
        })
      );
      req.reset();
    },
  });
};
