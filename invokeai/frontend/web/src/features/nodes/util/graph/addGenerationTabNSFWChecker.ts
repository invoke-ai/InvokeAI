import type { Graph } from 'features/nodes/util/graph/Graph';
import type { Invocation } from 'services/api/types';

import { NSFW_CHECKER } from './constants';

/**
 * Adds the NSFW checker to the output image
 * @param g The graph
 * @param imageOutput The current image output node
 * @returns The nsfw checker node
 */
export const addGenerationTabNSFWChecker = (
  g: Graph,
  imageOutput: Invocation<'l2i'> | Invocation<'img_nsfw'> | Invocation<'img_watermark'>
): Invocation<'img_nsfw'> => {
  const nsfw = g.addNode({
    id: NSFW_CHECKER,
    type: 'img_nsfw',
    is_intermediate: imageOutput.is_intermediate,
    board: imageOutput.board,
    use_cache: false,
  });

  imageOutput.is_intermediate = true;
  imageOutput.use_cache = true;
  imageOutput.board = undefined;

  g.addEdge(imageOutput, 'image', nsfw, 'image');

  return nsfw;
};
