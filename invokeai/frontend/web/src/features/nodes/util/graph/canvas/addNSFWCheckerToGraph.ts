import type { RootState } from 'app/store/store';
import { LATENTS_TO_IMAGE, NSFW_CHECKER } from 'features/nodes/util/graph/constants';
import { getBoardField, getIsIntermediate } from 'features/nodes/util/graph/graphBuilderUtils';
import type { Invocation, NonNullableGraph } from 'services/api/types';

export const addNSFWCheckerToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  nodeIdToAddTo = LATENTS_TO_IMAGE
): void => {
  const nodeToAddTo = graph.nodes[nodeIdToAddTo] as Invocation<'l2i'> | undefined;

  if (!nodeToAddTo) {
    // something has gone terribly awry
    return;
  }

  nodeToAddTo.is_intermediate = true;
  nodeToAddTo.use_cache = true;

  const nsfwCheckerNode: Invocation<'img_nsfw'> = {
    id: NSFW_CHECKER,
    type: 'img_nsfw',
    is_intermediate: getIsIntermediate(state),
    board: getBoardField(state),
  };

  graph.nodes[NSFW_CHECKER] = nsfwCheckerNode;
  graph.edges.push({
    source: {
      node_id: nodeIdToAddTo,
      field: 'image',
    },
    destination: {
      node_id: NSFW_CHECKER,
      field: 'image',
    },
  });
};
