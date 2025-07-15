import { objectEquals } from '@observ33r/object-equals';
import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getOriginalAndScaledSizesForTextToImage } from 'features/nodes/util/graph/graphBuilderUtils';
import type { DenoiseLatentsNodes, LatentToImageNodes } from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';
import { assert } from 'tsafe';

type AddTextToImageArg = {
  g: Graph;
  state: RootState;
  noise?: Invocation<'noise'>;
  denoise: Invocation<DenoiseLatentsNodes>;
  l2i: Invocation<LatentToImageNodes>;
};

export const addTextToImage = ({
  g,
  state,
  noise,
  denoise,
  l2i,
}: AddTextToImageArg): Invocation<'img_resize' | 'l2i' | 'flux_vae_decode' | 'sd3_l2i' | 'cogview4_l2i'> => {
  // Only set denoising values if they haven't been set already (e.g., by refiner)
  if (denoise.denoising_start === undefined) {
    denoise.denoising_start = 0;
  }
  if (denoise.denoising_end === undefined) {
    denoise.denoising_end = 1;
  }

  const { originalSize, scaledSize } = getOriginalAndScaledSizesForTextToImage(state);

  if (denoise.type === 'cogview4_denoise' || denoise.type === 'flux_denoise' || denoise.type === 'sd3_denoise') {
    denoise.width = scaledSize.width;
    denoise.height = scaledSize.height;
  } else {
    assert(denoise.type === 'denoise_latents');
    assert(noise, 'SD1.5/SD2/SDXL graphs require a noise node to be passed in');
    noise.width = scaledSize.width;
    noise.height = scaledSize.height;
  }

  g.upsertMetadata({
    width: originalSize.width,
    height: originalSize.height,
  });

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
