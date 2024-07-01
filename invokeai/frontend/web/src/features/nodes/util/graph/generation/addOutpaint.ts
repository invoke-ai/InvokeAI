import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import type { CanvasV2State, Size } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getInfill } from 'features/nodes/util/graph/graphBuilderUtils';
import type { ParameterPrecision } from 'features/parameters/types/parameterSchemas';
import { isEqual, pick } from 'lodash-es';
import type { Invocation } from 'services/api/types';

export const addOutpaint = async (
  g: Graph,
  manager: KonvaNodeManager,
  l2i: Invocation<'l2i'>,
  denoise: Invocation<'denoise_latents'>,
  vaeSource: Invocation<'main_model_loader' | 'sdxl_model_loader' | 'seamless' | 'vae_loader'>,
  modelLoader: Invocation<'main_model_loader' | 'sdxl_model_loader'>,
  originalSize: Size,
  scaledSize: Size,
  bbox: CanvasV2State['bbox'],
  compositing: CanvasV2State['compositing'],
  denoising_start: number,
  vaePrecision: ParameterPrecision
): Promise<Invocation<'canvas_paste_back'>> => {
  denoise.denoising_start = denoising_start;

  const cropBbox = pick(bbox, ['x', 'y', 'width', 'height']);
  const initialImage = await manager.getImageSourceImage({ bbox: cropBbox });
  const maskImage = await manager.getInpaintMaskImage({ bbox: cropBbox });
  const infill = getInfill(g, compositing);

  if (!isEqual(scaledSize, originalSize)) {
    // Scale before processing requires some resizing

    // Combine the inpaint mask and the initial image's alpha channel into a single mask
    const maskAlphaToMask = g.addNode({
      id: 'alpha_to_mask',
      type: 'tomask',
      image: { image_name: maskImage.image_name },
      invert: true,
    });
    const initialImageAlphaToMask = g.addNode({
      id: 'image_alpha_to_mask',
      type: 'tomask',
      image: { image_name: initialImage.image_name },
    });
    const maskCombine = g.addNode({
      id: 'mask_combine',
      type: 'mask_combine',
    });
    g.addEdge(maskAlphaToMask, 'image', maskCombine, 'mask1');
    g.addEdge(initialImageAlphaToMask, 'image', maskCombine, 'mask2');

    // Resize the combined and initial image to the scaled size
    const resizeInputMaskToScaledSize = g.addNode({
      id: 'resize_mask_to_scaled_size',
      type: 'img_resize',
      ...scaledSize,
    });
    g.addEdge(maskCombine, 'image', resizeInputMaskToScaledSize, 'image');

    // Resize the initial image to the scaled size and infill
    const resizeInputImageToScaledSize = g.addNode({
      id: 'resize_image_to_scaled_size',
      type: 'img_resize',
      image: { image_name: initialImage.image_name },
      ...scaledSize,
    });
    g.addEdge(resizeInputImageToScaledSize, 'image', infill, 'image');

    // Create the gradient denoising mask from the combined mask
    const createGradientMask = g.addNode({
      id: 'create_gradient_mask',
      type: 'create_gradient_mask',
      coherence_mode: compositing.canvasCoherenceMode,
      minimum_denoise: compositing.canvasCoherenceMinDenoise,
      edge_radius: compositing.canvasCoherenceEdgeSize,
      fp32: vaePrecision === 'fp32',
    });
    g.addEdge(infill, 'image', createGradientMask, 'image');
    g.addEdge(resizeInputMaskToScaledSize, 'image', createGradientMask, 'mask');
    g.addEdge(vaeSource, 'vae', createGradientMask, 'vae');
    g.addEdge(modelLoader, 'unet', createGradientMask, 'unet');
    g.addEdge(createGradientMask, 'denoise_mask', denoise, 'denoise_mask');

    // Decode infilled image and connect to denoise
    const i2l = g.addNode({ id: 'i2l', type: 'i2l' });
    g.addEdge(infill, 'image', i2l, 'image');
    g.addEdge(vaeSource, 'vae', i2l, 'vae');
    g.addEdge(i2l, 'latents', denoise, 'latents');

    // Resize the output image back to the original size
    const resizeOutputImageToOriginalSize = g.addNode({
      id: 'resize_image_to_original_size',
      type: 'img_resize',
      ...originalSize,
    });
    const resizeOutputMaskToOriginalSize = g.addNode({
      id: 'resize_mask_to_original_size',
      type: 'img_resize',
      ...originalSize,
    });
    const canvasPasteBack = g.addNode({
      id: 'canvas_paste_back',
      type: 'canvas_paste_back',
      mask_blur: compositing.maskBlur,
      source_image: { image_name: initialImage.image_name },
    });

    // Resize initial image and mask to scaled size, feed into to gradient mask

    // After denoising, resize the image and mask back to original size
    g.addEdge(l2i, 'image', resizeOutputImageToOriginalSize, 'image');
    g.addEdge(createGradientMask, 'expanded_mask_area', resizeOutputMaskToOriginalSize, 'image');

    // Finally, paste the generated masked image back onto the original image
    g.addEdge(resizeOutputImageToOriginalSize, 'image', canvasPasteBack, 'target_image');
    g.addEdge(resizeOutputMaskToOriginalSize, 'image', canvasPasteBack, 'mask');

    return canvasPasteBack;
  } else {
    infill.image = { image_name: initialImage.image_name };
    // No scale before processing, much simpler
    const i2l = g.addNode({ id: 'i2l', type: 'i2l' });
    const maskAlphaToMask = g.addNode({
      id: 'mask_alpha_to_mask',
      type: 'tomask',
      image: { image_name: maskImage.image_name },
      invert: true,
    });
    const initialImageAlphaToMask = g.addNode({
      id: 'image_alpha_to_mask',
      type: 'tomask',
      image: { image_name: initialImage.image_name },
    });
    const maskCombine = g.addNode({
      id: 'mask_combine',
      type: 'mask_combine',
    });
    const createGradientMask = g.addNode({
      id: 'create_gradient_mask',
      type: 'create_gradient_mask',
      coherence_mode: compositing.canvasCoherenceMode,
      minimum_denoise: compositing.canvasCoherenceMinDenoise,
      edge_radius: compositing.canvasCoherenceEdgeSize,
      fp32: vaePrecision === 'fp32',
      image: { image_name: initialImage.image_name },
    });
    const canvasPasteBack = g.addNode({
      id: 'canvas_paste_back',
      type: 'canvas_paste_back',
      mask_blur: compositing.maskBlur,
    });
    g.addEdge(maskAlphaToMask, 'image', maskCombine, 'mask1');
    g.addEdge(initialImageAlphaToMask, 'image', maskCombine, 'mask2');
    g.addEdge(maskCombine, 'image', createGradientMask, 'mask');
    g.addEdge(infill, 'image', i2l, 'image');
    g.addEdge(i2l, 'latents', denoise, 'latents');
    g.addEdge(vaeSource, 'vae', i2l, 'vae');
    g.addEdge(vaeSource, 'vae', createGradientMask, 'vae');
    g.addEdge(modelLoader, 'unet', createGradientMask, 'unet');
    g.addEdge(createGradientMask, 'denoise_mask', denoise, 'denoise_mask');
    g.addEdge(createGradientMask, 'expanded_mask_area', canvasPasteBack, 'mask');
    g.addEdge(infill, 'image', canvasPasteBack, 'source_image');
    g.addEdge(l2i, 'image', canvasPasteBack, 'target_image');

    return canvasPasteBack;
  }
};
