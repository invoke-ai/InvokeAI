import { objectEquals } from '@observ33r/object-equals';
import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasState, Dimensions } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getDenoisingStartAndEnd } from 'features/nodes/util/graph/graphBuilderUtils';
import type {
  DenoiseLatentsNodes,
  LatentToImageNodes,
  MainModelLoaderNodes,
  VaeSourceNodes,
} from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';

type AddImageToImageArg = {
  g: Graph;
  state: RootState;
  manager: CanvasManager;
  l2i: Invocation<LatentToImageNodes>;
  i2l: Invocation<'i2l' | 'flux_vae_encode' | 'sd3_i2l' | 'cogview4_i2l'>;
  denoise: Invocation<DenoiseLatentsNodes>;
  vaeSource: Invocation<VaeSourceNodes | MainModelLoaderNodes>;
  originalSize: Dimensions;
  scaledSize: Dimensions;
  bbox: CanvasState['bbox'];
};

export const addImageToImage = async ({
  g,
  state,
  manager,
  l2i,
  i2l,
  denoise,
  vaeSource,
  originalSize,
  scaledSize,
  bbox,
}: AddImageToImageArg): Promise<Invocation<'img_resize' | 'l2i' | 'flux_vae_decode' | 'sd3_l2i' | 'cogview4_l2i'>> => {
  const { denoising_start, denoising_end } = getDenoisingStartAndEnd(state);
  denoise.denoising_start = denoising_start;
  denoise.denoising_end = denoising_end;

  const adapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
  const { image_name } = await manager.compositor.getCompositeImageDTO(adapters, bbox.rect, {
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

    i2l.image = { image_name };

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
