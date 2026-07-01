import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { ImageOutputNodes } from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';

/**
 * Adds a watermark to the output image
 * @param g The graph
 * @param imageOutput The image output node
 * @returns The watermark node
 */
export const addWatermarker = (g: Graph, imageOutput: Invocation<ImageOutputNodes>): Invocation<'img_watermark'> => {
  const watermark = g.addNode({
    type: 'img_watermark',
    id: getPrefixedId('watermarker'),
  });

  g.addEdge(imageOutput, 'image', watermark, 'image');

  return watermark;
};
