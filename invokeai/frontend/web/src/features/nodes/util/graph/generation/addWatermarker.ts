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
  imageOutput: Invocation<'l2i' | 'img_nsfw' | 'img_watermark' | 'img_resize' | 'canvas_paste_back'>
): Invocation<'img_watermark'> => {
  const watermark = g.addNode({
    id: WATERMARKER,
    type: 'img_watermark',
  });

  g.addEdge(imageOutput, 'image', watermark, 'image');

  return watermark;
};
