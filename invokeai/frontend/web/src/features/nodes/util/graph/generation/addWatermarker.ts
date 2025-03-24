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
    | 'l2i'
    | 'img_nsfw'
    | 'img_watermark'
    | 'img_resize'
    | 'invokeai_img_blend'
    | 'apply_mask_to_image'
    | 'flux_vae_decode'
    | 'sd3_l2i'
  >
): Invocation<'img_watermark'> => {
  const watermark = g.addNode({
    type: 'img_watermark',
    id: getPrefixedId('watermarker'),
  });

  g.addEdge(imageOutput, 'image', watermark, 'image');

  return watermark;
};
