import { WATERMARKER } from 'features/nodes/util/graph/constants';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation } from 'services/api/types';

/**
 * Adds a watermark to the output image
 * @param g The graph
 * @param imageOutput The image output node
 * @returns The watermark node
 */
export const addWatermarker = (
  g: Graph,
  imageOutput: Invocation<'l2i'> | Invocation<'img_nsfw'> | Invocation<'img_watermark'> | Invocation<'img_resize'>
): Invocation<'img_watermark'> => {
  const watermark = g.addNode({
    id: WATERMARKER,
    type: 'img_watermark',
    is_intermediate: imageOutput.is_intermediate,
    board: imageOutput.board,
    use_cache: false,
  });

  // The watermarker node is the new image output - make the previous one intermediate
  imageOutput.is_intermediate = true;
  imageOutput.use_cache = true;
  imageOutput.board = undefined;

  g.addEdge(imageOutput, 'image', watermark, 'image');

  return watermark;
};
