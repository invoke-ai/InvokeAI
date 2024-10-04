import { getPrefixedId } from 'features/controlLayers/konva/util';
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
  imageOutput: Invocation<
    'l2i' | 'img_nsfw' | 'img_watermark' | 'img_resize' | 'canvas_v2_mask_and_crop' | 'flux_vae_decode'
  >
): Invocation<'img_watermark'> => {
  const watermark = g.addNode({
    type: 'img_watermark',
    id: getPrefixedId('watermarker'),
  });

  g.addEdge(imageOutput, 'image', watermark, 'image');

  return watermark;
};
