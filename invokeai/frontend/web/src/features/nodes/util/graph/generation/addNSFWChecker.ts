import { NSFW_CHECKER } from 'features/nodes/util/graph/constants';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation } from 'services/api/types';

/**
 * Adds the NSFW checker to the output image
 * @param g The graph
 * @param imageOutput The current image output node
 * @returns The nsfw checker node
 */
export const addNSFWChecker = (
  g: Graph,
  imageOutput: Invocation<'l2i'> | Invocation<'img_nsfw'> | Invocation<'img_watermark'> | Invocation<'img_resize'>
): Invocation<'img_nsfw'> => {
  const nsfw = g.addNode({
    id: NSFW_CHECKER,
    type: 'img_nsfw',
    is_intermediate: imageOutput.is_intermediate,
    board: imageOutput.board,
    use_cache: false,
  });

  // The NSFW checker node is the new image output - make the previous one intermediate
  imageOutput.is_intermediate = true;
  imageOutput.use_cache = true;
  imageOutput.board = undefined;

  g.addEdge(imageOutput, 'image', nsfw, 'image');

  return nsfw;
};
