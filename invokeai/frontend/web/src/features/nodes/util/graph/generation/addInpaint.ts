import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Dimensions } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { isEqual } from 'lodash-es';
import type { Invocation } from 'services/api/types';

export const addInpaint = async (
  state: RootState,
  g: Graph,
  manager: CanvasManager,
  l2i: Invocation<'l2i'>,
  denoise: Invocation<'denoise_latents'>,
  vaeSource: Invocation<'main_model_loader' | 'sdxl_model_loader' | 'seamless' | 'vae_loader'>,
  modelLoader: Invocation<'main_model_loader' | 'sdxl_model_loader'>,
  originalSize: Dimensions,
  scaledSize: Dimensions,
  denoising_start: number,
  fp32: boolean
): Promise<Invocation<'canvas_v2_mask_and_crop'>> => {
  denoise.denoising_start = denoising_start;

  const { params, canvasV2 } = state;
  const { bbox, session } = canvasV2;
  const { mode } = session;

  const initialImage = await manager.compositor.getCompositeRasterLayerImageDTO(bbox.rect);
  const maskImage = await manager.compositor.getCompositeInpaintMaskImageDTO(bbox.rect);

  if (!isEqual(scaledSize, originalSize)) {
    // Scale before processing requires some resizing
    const i2l = g.addNode({ id: getPrefixedId('i2l'), type: 'i2l', fp32 });
    const resizeImageToScaledSize = g.addNode({
      type: 'img_resize',
      id: getPrefixedId('resize_image_to_scaled_size'),
      image: { image_name: initialImage.image_name },
      ...scaledSize,
    });
    const alphaToMask = g.addNode({
      id: getPrefixedId('alpha_to_mask'),
      type: 'tomask',
      image: { image_name: maskImage.image_name },
      invert: true,
    });
    const resizeMaskToScaledSize = g.addNode({
      id: getPrefixedId('resize_mask_to_scaled_size'),
      type: 'img_resize',
      ...scaledSize,
    });
    const resizeImageToOriginalSize = g.addNode({
      id: getPrefixedId('resize_image_to_original_size'),
      type: 'img_resize',
      ...originalSize,
    });
    const resizeMaskToOriginalSize = g.addNode({
      id: getPrefixedId('resize_mask_to_original_size'),
      type: 'img_resize',
      ...originalSize,
    });
    const createGradientMask = g.addNode({
      id: getPrefixedId('create_gradient_mask'),
      type: 'create_gradient_mask',
      coherence_mode: params.canvasCoherenceMode,
      minimum_denoise: params.canvasCoherenceMinDenoise,
      edge_radius: params.canvasCoherenceEdgeSize,
      fp32,
    });
    const canvasPasteBack = g.addNode({
      id: getPrefixedId('canvas_v2_mask_and_crop'),
      type: 'canvas_v2_mask_and_crop',
      mask_blur: params.maskBlur,
    });

    // Resize initial image and mask to scaled size, feed into to gradient mask
    g.addEdge(alphaToMask, 'image', resizeMaskToScaledSize, 'image');
    g.addEdge(resizeImageToScaledSize, 'image', i2l, 'image');
    g.addEdge(i2l, 'latents', denoise, 'latents');
    g.addEdge(vaeSource, 'vae', i2l, 'vae');

    g.addEdge(vaeSource, 'vae', createGradientMask, 'vae');
    g.addEdge(modelLoader, 'unet', createGradientMask, 'unet');
    g.addEdge(resizeImageToScaledSize, 'image', createGradientMask, 'image');
    g.addEdge(resizeMaskToScaledSize, 'image', createGradientMask, 'mask');

    g.addEdge(createGradientMask, 'denoise_mask', denoise, 'denoise_mask');

    // After denoising, resize the image and mask back to original size
    g.addEdge(l2i, 'image', resizeImageToOriginalSize, 'image');
    g.addEdge(createGradientMask, 'expanded_mask_area', resizeMaskToOriginalSize, 'image');

    // Finally, paste the generated masked image back onto the original image
    g.addEdge(resizeImageToOriginalSize, 'image', canvasPasteBack, 'generated_image');
    g.addEdge(resizeMaskToOriginalSize, 'image', canvasPasteBack, 'mask');

    if (mode === 'generate') {
      canvasPasteBack.source_image = { image_name: initialImage.image_name };
    }

    return canvasPasteBack;
  } else {
    // No scale before processing, much simpler
    const i2l = g.addNode({
      id: getPrefixedId('i2l'),
      type: 'i2l',
      image: { image_name: initialImage.image_name },
      fp32,
    });
    const alphaToMask = g.addNode({
      id: getPrefixedId('alpha_to_mask'),
      type: 'tomask',
      image: { image_name: maskImage.image_name },
      invert: true,
    });
    const createGradientMask = g.addNode({
      id: getPrefixedId('create_gradient_mask'),
      type: 'create_gradient_mask',
      coherence_mode: params.canvasCoherenceMode,
      minimum_denoise: params.canvasCoherenceMinDenoise,
      edge_radius: params.canvasCoherenceEdgeSize,
      fp32,
      image: { image_name: initialImage.image_name },
    });
    const canvasPasteBack = g.addNode({
      id: getPrefixedId('canvas_v2_mask_and_crop'),
      type: 'canvas_v2_mask_and_crop',
      mask_blur: params.maskBlur,
    });

    g.addEdge(alphaToMask, 'image', createGradientMask, 'mask');
    g.addEdge(i2l, 'latents', denoise, 'latents');
    g.addEdge(vaeSource, 'vae', i2l, 'vae');
    g.addEdge(vaeSource, 'vae', createGradientMask, 'vae');
    g.addEdge(modelLoader, 'unet', createGradientMask, 'unet');
    g.addEdge(createGradientMask, 'denoise_mask', denoise, 'denoise_mask');
    g.addEdge(createGradientMask, 'expanded_mask_area', canvasPasteBack, 'mask');

    g.addEdge(l2i, 'image', canvasPasteBack, 'generated_image');

    if (mode === 'generate') {
      canvasPasteBack.source_image = { image_name: initialImage.image_name };
    }

    return canvasPasteBack;
  }
};
