import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { ImageOutputNodes } from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';

/**
 * Adds the NSFW checker to the output image
 * @param g The graph
 * @param imageOutput The current image output node
 * @returns The nsfw checker node
 */
export const addNSFWChecker = (g: Graph, imageOutput: Invocation<ImageOutputNodes>): Invocation<'img_nsfw'> => {
  const nsfw = g.addNode({
    type: 'img_nsfw',
    id: getPrefixedId('nsfw_checker'),
  });

  g.addEdge(imageOutput, 'image', nsfw, 'image');

  return nsfw;
};
