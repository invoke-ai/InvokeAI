import { logger } from 'app/logging/logger';
import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { ImageField } from 'features/nodes/types/common';
import {
  isFloatFieldCollectionInputInstance,
  isImageFieldCollectionInputInstance,
  isIntegerFieldCollectionInputInstance,
  isStringFieldCollectionInputInstance,
} from 'features/nodes/types/field';
import type { InvocationNodeEdge } from 'features/nodes/types/invocation';
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

      const addBatchDataCollectionItem = (edges: InvocationNodeEdge[], items?: ImageField[] | string[] | number[]) => {
        const batchDataCollectionItem: NonNullable<Batch['data']>[number] = [];
        for (const edge of edges) {
          if (!edge.targetHandle) {
            break;
          }
          batchDataCollectionItem.push({
            node_path: edge.target,
            field_name: edge.targetHandle,
            items,
          });
        }
        if (batchDataCollectionItem.length > 0) {
          data.push(batchDataCollectionItem);
        }
      };

      // Grab image batch nodes for special handling
      const imageBatchNodes = nodes.nodes.filter(isInvocationNode).filter((node) => node.data.type === 'image_batch');

      for (const node of imageBatchNodes) {
        // Satisfy TS
        const images = node.data.inputs['images'];
        if (!isImageFieldCollectionInputInstance(images)) {
          log.warn({ nodeId: node.id }, 'Image batch images field is not an image collection');
          break;
        }

        // Find outgoing edges from the batch node, we will remove these from the graph and create batch data collection items from them instead
        const edgesFromImageBatch = nodes.edges.filter((e) => e.source === node.id && e.sourceHandle === 'image');
        addBatchDataCollectionItem(edgesFromImageBatch, images.value);
      }

      // Grab string batch nodes for special handling
      const stringBatchNodes = nodes.nodes.filter(isInvocationNode).filter((node) => node.data.type === 'string_batch');
      for (const node of stringBatchNodes) {
        // Satisfy TS
        const strings = node.data.inputs['strings'];
        if (!isStringFieldCollectionInputInstance(strings)) {
          log.warn({ nodeId: node.id }, 'String batch strings field is not a string collection');
          break;
        }

        // Find outgoing edges from the batch node, we will remove these from the graph and create batch data collection items from them instead
        const edgesFromStringBatch = nodes.edges.filter((e) => e.source === node.id && e.sourceHandle === 'value');
        addBatchDataCollectionItem(edgesFromStringBatch, strings.value);
      }

      // Grab integer batch nodes for special handling
      const integerBatchNodes = nodes.nodes
        .filter(isInvocationNode)
        .filter((node) => node.data.type === 'integer_batch');
      for (const node of integerBatchNodes) {
        // Satisfy TS
        const integers = node.data.inputs['integers'];
        if (!isIntegerFieldCollectionInputInstance(integers)) {
          log.warn({ nodeId: node.id }, 'Integer batch integers field is not an integer collection');
          break;
        }
        if (!integers.value) {
          log.warn({ nodeId: node.id }, 'Integer batch integers field is empty');
          break;
        }

        // Find outgoing edges from the batch node, we will remove these from the graph and create batch data collection items from them instead
        const edgesFromStringBatch = nodes.edges.filter((e) => e.source === node.id && e.sourceHandle === 'value');
        addBatchDataCollectionItem(edgesFromStringBatch, integers.value);
      }

      // Grab float batch nodes for special handling
      const floatBatchNodes = nodes.nodes.filter(isInvocationNode).filter((node) => node.data.type === 'float_batch');
      for (const node of floatBatchNodes) {
        // Satisfy TS
        const floats = node.data.inputs['floats'];
        if (!isFloatFieldCollectionInputInstance(floats)) {
          log.warn({ nodeId: node.id }, 'Float batch floats field is not a float collection');
          break;
        }
        if (!floats.value) {
          log.warn({ nodeId: node.id }, 'Float batch floats field is empty');
          break;
        }

        // Find outgoing edges from the batch node, we will remove these from the graph and create batch data collection items from them instead
        const edgesFromStringBatch = nodes.edges.filter((e) => e.source === node.id && e.sourceHandle === 'value');
        addBatchDataCollectionItem(edgesFromStringBatch, floats.value);
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
