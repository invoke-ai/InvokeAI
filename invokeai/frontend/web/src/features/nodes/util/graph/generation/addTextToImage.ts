import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Dimensions } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { isEqual } from 'lodash-es';
import type { Invocation } from 'services/api/types';

export const addTextToImage = (
  g: Graph,
  l2i: Invocation<'l2i' | 'flux_vae_decode'>,
  originalSize: Dimensions,
  scaledSize: Dimensions
): Invocation<'img_resize' | 'l2i' | 'flux_vae_decode'> => {
  if (!isEqual(scaledSize, originalSize)) {
    // We need to resize the output image back to the original size
    const resizeImageToOriginalSize = g.addNode({
      id: getPrefixedId('resize_image_to_original_size'),
      type: 'img_resize',
      ...originalSize,
    });
    g.addEdge(l2i, 'image', resizeImageToOriginalSize, 'image');

    return resizeImageToOriginalSize;
  } else {
    return l2i;
  }
};
