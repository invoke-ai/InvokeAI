import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import type { CanvasV2State, Dimensions } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { ParameterStrength } from 'features/parameters/types/parameterSchemas';
import { isEqual, pick } from 'lodash-es';
import type { Invocation } from 'services/api/types';

export const addImageToImage = async (
  g: Graph,
  manager: KonvaNodeManager,
  l2i: Invocation<'l2i'>,
  denoise: Invocation<'denoise_latents'>,
  vaeSource: Invocation<'main_model_loader' | 'seamless' | 'vae_loader'>,
  imageOutput: Invocation<'canvas_paste_back' | 'img_nsfw' | 'img_resize' | 'img_watermark' | 'l2i'>,
  originalSize: Dimensions,
  scaledSize: Dimensions,
  bbox: CanvasV2State['bbox'],
  strength: ParameterStrength
): Promise<Invocation<'img_resize' | 'l2i'>> => {
  denoise.denoising_start = 1 - strength;

  const cropBbox = pick(bbox, ['x', 'y', 'width', 'height']);
  const initialImage = await manager.util.getImageSourceImage({
    bbox: cropBbox,
    preview: true,
  });

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
