import type { Dimensions } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { isEqual } from 'lodash-es';
import type { Invocation } from 'services/api/types';

export const addTextToImage = (
  g: Graph,
  l2i: Invocation<'l2i'>,
  imageOutput: Invocation<'canvas_paste_back' | 'img_nsfw' | 'img_resize' | 'img_watermark' | 'l2i'>,
  originalSize: Dimensions,
  scaledSize: Dimensions
) => {
  if (!isEqual(scaledSize, originalSize)) {
    // We need to resize the output image back to the original size
    const resizeImageToOriginalSize = g.addNode({
      id: 'resize_image_to_original_size',
      type: 'img_resize',
      ...originalSize,
    });
    g.addEdge(l2i, 'image', resizeImageToOriginalSize, 'image');

    // This is the new output node
    imageOutput = resizeImageToOriginalSize;
  }
};
