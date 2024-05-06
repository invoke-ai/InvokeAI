import type { RootState } from 'app/store/store';
import { isInitialImageLayer } from 'features/controlLayers/store/controlLayersSlice';
import type { ImageField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/Graph';
import { MetadataUtil } from 'features/nodes/util/graph/MetadataUtil';
import type { Invocation } from 'services/api/types';

import { IMAGE_TO_LATENTS, RESIZE } from './constants';

/**
 * Adds the initial image to the graph and connects it to the denoise and noise nodes.
 * @param state The current Redux state
 * @param g The graph to add the initial image to
 * @param denoise The denoise node in the graph
 * @param noise The noise node in the graph
 * @returns Whether the initial image was added to the graph
 */
export const addGenerationTabInitialImage = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  noise: Invocation<'noise'>
): Invocation<'i2l'> | null => {
  // Remove Existing UNet Connections
  const { img2imgStrength, vaePrecision, model } = state.generation;
  const { refinerModel, refinerStart } = state.sdxl;
  const { width, height } = state.controlLayers.present.size;
  const initialImageLayer = state.controlLayers.present.layers.find(isInitialImageLayer);
  const initialImage = initialImageLayer?.isEnabled ? initialImageLayer?.image : null;

  if (!initialImage) {
    return null;
  }

  const isSDXL = model?.base === 'sdxl';
  const useRefinerStartEnd = isSDXL && Boolean(refinerModel);
  const image: ImageField = {
    image_name: initialImage.imageName,
  };

  denoise.denoising_start = useRefinerStartEnd ? Math.min(refinerStart, 1 - img2imgStrength) : 1 - img2imgStrength;
  denoise.denoising_end = useRefinerStartEnd ? refinerStart : 1;

  const i2l = g.addNode({
    type: 'i2l',
    id: IMAGE_TO_LATENTS,
    fp32: vaePrecision === 'fp32',
  });
  g.addEdge(i2l, 'latents', denoise, 'latents');

  if (initialImage.width !== width || initialImage.height !== height) {
    // The init image needs to be resized to the specified width and height before being passed to `IMAGE_TO_LATENTS`
    const resize = g.addNode({
      id: RESIZE,
      type: 'img_resize',
      image,
      width,
      height,
    });
    // The `RESIZE` node then passes its image, to `IMAGE_TO_LATENTS`
    g.addEdge(resize, 'image', i2l, 'image');
    // The `RESIZE` node also passes its width and height to `NOISE`
    g.addEdge(resize, 'width', noise, 'width');
    g.addEdge(resize, 'height', noise, 'height');
  } else {
    // We are not resizing, so we need to set the image on the `IMAGE_TO_LATENTS` node explicitly
    i2l.image = image;
    g.addEdge(i2l, 'width', noise, 'width');
    g.addEdge(i2l, 'height', noise, 'height');
  }

  MetadataUtil.add(g, {
    generation_mode: isSDXL ? 'sdxl_img2img' : 'img2img',
    strength: img2imgStrength,
    init_image: initialImage.imageName,
  });

  return i2l;
};
