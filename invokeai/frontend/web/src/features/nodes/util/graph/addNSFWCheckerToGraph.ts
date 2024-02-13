import type { RootState } from 'app/store/store';
import type { ImageNSFWBlurInvocation, LatentsToImageInvocation, NonNullableGraph } from 'services/api/types';

import { LATENTS_TO_IMAGE, NSFW_CHECKER } from './constants';

export const addNSFWCheckerToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  nodeIdToAddTo = LATENTS_TO_IMAGE
): void => {
  const nodeToAddTo = graph.nodes[nodeIdToAddTo] as LatentsToImageInvocation | undefined;

  if (!nodeToAddTo) {
    // something has gone terribly awry
    return;
  }

  nodeToAddTo.is_intermediate = true;
  nodeToAddTo.use_cache = true;

  const nsfwCheckerNode: ImageNSFWBlurInvocation = {
    id: NSFW_CHECKER,
    type: 'img_nsfw',
    is_intermediate: true,
  };

  graph.nodes[NSFW_CHECKER] = nsfwCheckerNode as ImageNSFWBlurInvocation;
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
