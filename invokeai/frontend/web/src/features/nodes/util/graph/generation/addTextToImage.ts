import { objectEquals } from '@observ33r/object-equals';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Dimensions } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { DenoiseLatentsNodes, LatentToImageNodes } from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';

type AddTextToImageArg = {
  g: Graph;
  denoise: Invocation<DenoiseLatentsNodes>;
  l2i: Invocation<LatentToImageNodes>;
  originalSize: Dimensions;
  scaledSize: Dimensions;
};

export const addTextToImage = ({
  g,
  denoise,
  l2i,
  originalSize,
  scaledSize,
}: AddTextToImageArg): Invocation<'img_resize' | 'l2i' | 'flux_vae_decode' | 'sd3_l2i' | 'cogview4_l2i'> => {
  denoise.denoising_start = 0;
  denoise.denoising_end = 1;

  if (!objectEquals(scaledSize, originalSize)) {
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
