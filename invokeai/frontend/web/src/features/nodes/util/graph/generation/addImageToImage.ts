import { objectEquals } from '@observ33r/object-equals';
import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getDenoisingStartAndEnd,
  getOriginalAndScaledSizesForOtherModes,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type {
  DenoiseLatentsNodes,
  LatentToImageNodes,
  MainModelLoaderNodes,
  VaeSourceNodes,
} from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';
import { assert } from 'tsafe';

type AddImageToImageArg = {
  g: Graph;
  state: RootState;
  manager: CanvasManager;
  l2i: Invocation<LatentToImageNodes>;
  i2l: Invocation<'i2l' | 'flux_vae_encode' | 'sd3_i2l' | 'cogview4_i2l' | 'z_image_i2l'>;
  noise?: Invocation<'noise'>;
  denoise: Invocation<DenoiseLatentsNodes>;
  vaeSource: Invocation<VaeSourceNodes | MainModelLoaderNodes>;
};

export const addImageToImage = async ({
  g,
  state,
  manager,
  l2i,
  i2l,
  noise,
  denoise,
  vaeSource,
}: AddImageToImageArg): Promise<
  Invocation<'img_resize' | 'l2i' | 'flux_vae_decode' | 'sd3_l2i' | 'cogview4_l2i' | 'z_image_l2i'>
> => {
  const { denoising_start, denoising_end } = getDenoisingStartAndEnd(state);
  denoise.denoising_start = denoising_start;
  denoise.denoising_end = denoising_end;

  const { originalSize, scaledSize, rect } = getOriginalAndScaledSizesForOtherModes(state);

  if (
    denoise.type === 'cogview4_denoise' ||
    denoise.type === 'flux_denoise' ||
    denoise.type === 'sd3_denoise' ||
    denoise.type === 'z_image_denoise'
  ) {
    denoise.width = scaledSize.width;
    denoise.height = scaledSize.height;
  } else {
    assert(denoise.type === 'denoise_latents');
    assert(noise, 'SD1.5/SD2/SDXL graphs require a noise node to be passed in');
    noise.width = scaledSize.width;
    noise.height = scaledSize.height;
  }

  const adapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
  const { image_name } = await manager.compositor.getCompositeImageDTO(adapters, rect, {
    is_intermediate: true,
    silent: true,
  });

  if (!objectEquals(scaledSize, originalSize)) {
    // Resize the initial image to the scaled size, denoise, then resize back to the original size
    const resizeImageToScaledSize = g.addNode({
      type: 'img_resize',
      id: getPrefixedId('initial_image_resize_in'),
      image: { image_name },
      ...scaledSize,
    });

    const resizeImageToOriginalSize = g.addNode({
      type: 'img_resize',
      id: getPrefixedId('initial_image_resize_out'),
      ...originalSize,
    });

    g.addEdge(vaeSource, 'vae', i2l, 'vae');
    g.addEdge(resizeImageToScaledSize, 'image', i2l, 'image');
    g.addEdge(i2l, 'latents', denoise, 'latents');
    g.addEdge(l2i, 'image', resizeImageToOriginalSize, 'image');

    // This is the new output node
    return resizeImageToOriginalSize;
  } else {
    // No need to resize, just decode
    i2l.image = { image_name };
    g.addEdge(vaeSource, 'vae', i2l, 'vae');
    g.addEdge(i2l, 'latents', denoise, 'latents');
    return l2i;
  }
};
