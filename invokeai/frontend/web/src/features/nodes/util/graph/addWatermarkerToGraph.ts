import type { RootState } from 'app/store/store';
import type {
  ImageNSFWBlurInvocation,
  ImageWatermarkInvocation,
  LatentsToImageInvocation,
  NonNullableGraph,
} from 'services/api/types';

import { LATENTS_TO_IMAGE, NSFW_CHECKER, WATERMARKER } from './constants';
import { getBoardField, getIsIntermediate } from './graphBuilderUtils';

export const addWatermarkerToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  nodeIdToAddTo = LATENTS_TO_IMAGE
): void => {
  const nodeToAddTo = graph.nodes[nodeIdToAddTo] as LatentsToImageInvocation | undefined;

  const nsfwCheckerNode = graph.nodes[NSFW_CHECKER] as ImageNSFWBlurInvocation | undefined;

  if (!nodeToAddTo) {
    // something has gone terribly awry
    return;
  }

  const watermarkerNode: ImageWatermarkInvocation = {
    id: WATERMARKER,
    type: 'img_watermark',
    is_intermediate: getIsIntermediate(state),
    board: getBoardField(state),
  };

  graph.nodes[WATERMARKER] = watermarkerNode;

  // no matter the situation, we want the l2i node to be intermediate
  nodeToAddTo.is_intermediate = true;
  nodeToAddTo.use_cache = true;

  if (nsfwCheckerNode) {
    // if we are using NSFW checker, we need to "disable" it output by marking it intermediate,
    // then connect it to the watermark node
    nsfwCheckerNode.is_intermediate = true;
    graph.edges.push({
      source: {
        node_id: NSFW_CHECKER,
        field: 'image',
      },
      destination: {
        node_id: WATERMARKER,
        field: 'image',
      },
    });
  } else {
    // otherwise we just connect to the watermark node
    graph.edges.push({
      source: {
        node_id: nodeIdToAddTo,
        field: 'image',
      },
      destination: {
        node_id: WATERMARKER,
        field: 'image',
      },
    });
  }
};
