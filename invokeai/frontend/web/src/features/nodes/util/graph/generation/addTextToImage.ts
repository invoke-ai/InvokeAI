import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Dimensions } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { CanvasOutputs } from 'features/nodes/util/graph/graphBuilderUtils';
import { isEqual } from 'lodash-es';
import type { Invocation } from 'services/api/types';

type AddTextToImageArg = {
  g: Graph;
  l2i: Invocation<'l2i' | 'flux_vae_decode'>;
  originalSize: Dimensions;
  scaledSize: Dimensions;
};

export const addTextToImage = ({ g, l2i, originalSize, scaledSize }: AddTextToImageArg): CanvasOutputs => {
  if (!isEqual(scaledSize, originalSize)) {
    // We need to resize the output image back to the original size
    const resizeImageToOriginalSize = g.addNode({
      id: getPrefixedId('resize_image_to_original_size'),
      type: 'img_resize',
      ...originalSize,
    });
    g.addEdge(l2i, 'image', resizeImageToOriginalSize, 'image');

    return { scaled: resizeImageToOriginalSize, unscaled: l2i };
  } else {
    return { scaled: l2i, unscaled: l2i };
  }
};
