import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasV2State, Size } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { isEqual, pick } from 'lodash-es';
import type { Invocation } from 'services/api/types';

export const addImageToImage = async (
  g: Graph,
  manager: CanvasManager,
  l2i: Invocation<'l2i'>,
  denoise: Invocation<'denoise_latents'>,
  vaeSource: Invocation<'main_model_loader' | 'sdxl_model_loader' | 'seamless' | 'vae_loader'>,
  originalSize: Size,
  scaledSize: Size,
  bbox: CanvasV2State['bbox'],
  denoising_start: number
): Promise<Invocation<'img_resize' | 'l2i'>> => {
  denoise.denoising_start = denoising_start;

  const cropBbox = pick(bbox, ['x', 'y', 'width', 'height']);
  const initialImage = await manager.getImageSourceImage({ bbox: cropBbox });

  if (!isEqual(scaledSize, originalSize)) {
    // Resize the initial image to the scaled size, denoise, then resize back to the original size
    const resizeImageToScaledSize = g.addNode({
      id: 'initial_image_resize_in',
      type: 'img_resize',
      image: { image_name: initialImage.image_name },
      ...scaledSize,
    });
    const i2l = g.addNode({ id: 'i2l', type: 'i2l' });
    const resizeImageToOriginalSize = g.addNode({
      id: 'initial_image_resize_out',
      type: 'img_resize',
      ...originalSize,
    });

    g.addEdge(vaeSource, 'vae', i2l, 'vae');
    g.addEdge(resizeImageToScaledSize, 'image', i2l, 'image');
    g.addEdge(i2l, 'latents', denoise, 'latents');
    g.addEdge(l2i, 'image', resizeImageToOriginalSize, 'image');

    // This is the new output node
    return resizeImageToOriginalSize;
  } else {
    // No need to resize, just denoise
    const i2l = g.addNode({ id: 'i2l', type: 'i2l', image: { image_name: initialImage.image_name } });
    g.addEdge(vaeSource, 'vae', i2l, 'vae');
    g.addEdge(i2l, 'latents', denoise, 'latents');
    return l2i;
  }
};
