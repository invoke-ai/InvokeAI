import { logger } from 'app/logging/logger';
import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { isImageFieldCollectionInputInstance } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { buildNodesGraph } from 'features/nodes/util/graph/buildNodesGraph';
import { buildWorkflowWithValidation } from 'features/nodes/util/workflow/buildWorkflow';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import type { Batch, BatchConfig } from 'services/api/types';

const log = logger('workflows');

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
      const imageBatchNodes = nodes.nodes.filter(isInvocationNode).filter((node) => node.data.type === 'image_batch');
      for (const node of imageBatchNodes) {
        const images = node.data.inputs['images'];
        if (!isImageFieldCollectionInputInstance(images)) {
          log.warn({ nodeId: node.id }, 'Image batch images field is not an image collection');
          break;
        }
        const edgesFromImageBatch = nodes.edges.filter((e) => e.source === node.id && e.sourceHandle === 'image');
        const batchDataCollectionItem: NonNullable<Batch['data']>[number] = [];
        for (const edge of edgesFromImageBatch) {
          if (!edge.targetHandle) {
            break;
          }
          batchDataCollectionItem.push({
            node_path: edge.target,
            field_name: edge.targetHandle,
            items: images.value,
          });
        }
        if (batchDataCollectionItem.length > 0) {
          data.push(batchDataCollectionItem);
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
