import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import {
  ImageNSFWBlurInvocation,
  LatentsToImageInvocation,
  MetadataAccumulatorInvocation,
} from 'services/api/types';
import {
  LATENTS_TO_IMAGE,
  METADATA_ACCUMULATOR,
  NSFW_CHECKER,
} from './constants';

export const addNSFWCheckerToGraph = (
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

  const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
    | MetadataAccumulatorInvocation
    | undefined;

  if (!nodeToAddTo) {
    // something has gone terribly awry
    return;
  }

  nodeToAddTo.is_intermediate = true;

  const nsfwCheckerNode: ImageNSFWBlurInvocation = {
    id: NSFW_CHECKER,
    type: 'img_nsfw',
    is_intermediate,
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

  if (metadataAccumulator) {
    graph.edges.push({
      source: {
        node_id: METADATA_ACCUMULATOR,
        field: 'metadata',
      },
      destination: {
        node_id: NSFW_CHECKER,
        field: 'metadata',
      },
    });
  }
};
