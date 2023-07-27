import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import {
  ImageNSFWBlurInvocation,
  ImageWatermarkInvocation,
  LatentsToImageInvocation,
  MetadataAccumulatorInvocation,
} from 'services/api/types';
import {
  LATENTS_TO_IMAGE,
  METADATA_ACCUMULATOR,
  NSFW_CHECKER,
  WATERMARKER,
} from './constants';

export const addWatermarkerToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  nodeIdToAddTo = LATENTS_TO_IMAGE
): void => {
  const activeTabName = activeTabNameSelector(state);

  const is_intermediate =
    activeTabName === 'unifiedCanvas' ? !state.canvas.shouldAutoSave : false;

  const nodeToAddTo = graph.nodes[nodeIdToAddTo] as
    | LatentsToImageInvocation
    | undefined;

  const nsfwCheckerNode = graph.nodes[NSFW_CHECKER] as
    | ImageNSFWBlurInvocation
    | undefined;

  const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
    | MetadataAccumulatorInvocation
    | undefined;

  if (!nodeToAddTo) {
    // something has gone terribly awry
    return;
  }

  const watermarkerNode: ImageWatermarkInvocation = {
    id: WATERMARKER,
    type: 'img_watermark',
    is_intermediate,
  };

  graph.nodes[WATERMARKER] = watermarkerNode;

  // no matter the situation, we want the l2i node to be intermediate
  nodeToAddTo.is_intermediate = true;

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

  if (metadataAccumulator) {
    graph.edges.push({
      source: {
        node_id: METADATA_ACCUMULATOR,
        field: 'metadata',
      },
      destination: {
        node_id: WATERMARKER,
        field: 'metadata',
      },
    });
  }
};
