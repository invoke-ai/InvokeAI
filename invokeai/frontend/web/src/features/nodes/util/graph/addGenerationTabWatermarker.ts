import type { Graph } from 'features/nodes/util/graph/Graph';
import type { Invocation } from 'services/api/types';

import { WATERMARKER } from './constants';

/**
 * Adds a watermark to the output image
 * @param g The graph
 * @param imageOutput The image output node
 * @returns The watermark node
 */
export const addGenerationTabWatermarker = (
  g: Graph,
  imageOutput: Invocation<'l2i'> | Invocation<'img_nsfw'> | Invocation<'img_watermark'>
): Invocation<'img_watermark'> => {
  const watermark = g.addNode({
    id: WATERMARKER,
    type: 'img_watermark',
    is_intermediate: imageOutput.is_intermediate,
    board: imageOutput.board,
    use_cache: false,
  });

  imageOutput.is_intermediate = true;
  imageOutput.use_cache = true;
  imageOutput.board = undefined;

  g.addEdge(imageOutput, 'image', watermark, 'image');

  return watermark;
};
