import { enqueueRequested } from 'app/store/actions';
import { buildNodesGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { buildWorkflowWithValidation } from 'features/nodes/util/workflow/buildWorkflow';
import { workflowBatchEnqueued } from 'features/progress/store/progressSlice';
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
      const builtWorkflow = buildWorkflowWithValidation({
        nodes,
        edges,
        workflow,
      });

      if (builtWorkflow) {
        // embedded workflows don't have an id
        delete builtWorkflow.id;
      }

      const batchConfig: BatchConfig = {
        batch: {
          graph,
          workflow: builtWorkflow,
          runs: state.generation.iterations,
        },
        prepend: action.payload.prepend,
      };

      try {
        const req = dispatch(
          queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
            fixedCacheKey: 'enqueueBatch',
          })
        );
        const enqueueResult = await req.unwrap();
        req.reset();
        if (enqueueResult.batch.batch_id) {
          dispatch(workflowBatchEnqueued(enqueueResult.batch.batch_id));
        }
      } catch {
        // no-op
      }
    },
  });
};
