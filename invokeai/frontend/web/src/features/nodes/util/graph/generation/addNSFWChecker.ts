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
  imageOutput: Invocation<'l2i' | 'img_nsfw' | 'img_watermark' | 'img_resize' | 'canvas_paste_back'>
): Invocation<'img_nsfw'> => {
  const nsfw = g.addNode({
    id: NSFW_CHECKER,
    type: 'img_nsfw',
  });

  g.addEdge(imageOutput, 'image', nsfw, 'image');

  return nsfw;
};
