import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasState, Dimensions } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type {
  DenoiseLatentsNodes,
  LatentToImageNodes,
  MainModelLoaderNodes,
  VaeSourceNodes,
} from 'features/nodes/util/graph/types';
import { isEqual } from 'lodash-es';
import type { Invocation } from 'services/api/types';

type AddImageToImageArg = {
  g: Graph;
  manager: CanvasManager;
  l2i: Invocation<LatentToImageNodes>;
  i2lNodeType: 'i2l' | 'flux_vae_encode' | 'sd3_i2l' | 'cogview4_i2l';
  denoise: Invocation<DenoiseLatentsNodes>;
  vaeSource: Invocation<VaeSourceNodes | MainModelLoaderNodes>;
  originalSize: Dimensions;
  scaledSize: Dimensions;
  bbox: CanvasState['bbox'];
  denoising_start: number;
  fp32: boolean;
};

export const addImageToImage = async ({
  g,
  manager,
  l2i,
  i2lNodeType,
  denoise,
  vaeSource,
  originalSize,
  scaledSize,
  bbox,
  denoising_start,
  fp32,
}: AddImageToImageArg): Promise<Invocation<'img_resize' | 'l2i' | 'flux_vae_decode' | 'sd3_l2i' | 'cogview4_l2i'>> => {
  denoise.denoising_start = denoising_start;
  const adapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
  const { image_name } = await manager.compositor.getCompositeImageDTO(adapters, bbox.rect, {
    is_intermediate: true,
    silent: true,
  });

  if (!isEqual(scaledSize, originalSize)) {
    // Resize the initial image to the scaled size, denoise, then resize back to the original size
    const resizeImageToScaledSize = g.addNode({
      type: 'img_resize',
      id: getPrefixedId('initial_image_resize_in'),
      image: { image_name },
      ...scaledSize,
    });

    const i2l = g.addNode({
      id: i2lNodeType,
      type: i2lNodeType,
      image: image_name ? { image_name } : undefined,
      ...(i2lNodeType === 'i2l' ? { fp32 } : {}),
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
    const i2l = g.addNode({
      id: i2lNodeType,
      type: i2lNodeType,
      image: image_name ? { image_name } : undefined,
      ...(i2lNodeType === 'i2l' ? { fp32 } : {}),
    });
    g.addEdge(vaeSource, 'vae', i2l, 'vae');
    g.addEdge(i2l, 'latents', denoise, 'latents');
    return l2i;
  }
};
