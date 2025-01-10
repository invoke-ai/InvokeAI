import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { Dimensions } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { isEqual } from 'lodash-es';
import type { Invocation } from 'services/api/types';

type AddInpaintArg = {
  state: RootState;
  g: Graph;
  manager: CanvasManager;
  l2i: Invocation<'l2i' | 'flux_vae_decode' | 'sd3_l2i'>;
  i2lNodeType: 'i2l' | 'flux_vae_encode' | 'sd3_i2l';
  denoise: Invocation<'denoise_latents' | 'flux_denoise' | 'sd3_denoise'>;
  vaeSource: Invocation<
    'main_model_loader' | 'sdxl_model_loader' | 'flux_model_loader' | 'seamless' | 'vae_loader' | 'sd3_model_loader'
  >;
  modelLoader: Invocation<'main_model_loader' | 'sdxl_model_loader' | 'flux_model_loader' | 'sd3_model_loader'>;
  originalSize: Dimensions;
  scaledSize: Dimensions;
  denoising_start: number;
  fp32: boolean;
};

export const addInpaint = async ({
  state,
  g,
  manager,
  l2i,
  i2lNodeType,
  denoise,
  vaeSource,
  modelLoader,
  originalSize,
  scaledSize,
  denoising_start,
  fp32,
}: AddInpaintArg): Promise<Invocation<'canvas_v2_mask_and_crop' | 'img_resize'>> => {
  denoise.denoising_start = denoising_start;

  const params = selectParamsSlice(state);
  const canvasSettings = selectCanvasSettingsSlice(state);
  const canvas = selectCanvasSlice(state);

  const { bbox } = canvas;

  const rasterAdapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
  const initialImage = await manager.compositor.getCompositeImageDTO(rasterAdapters, bbox.rect, {
    is_intermediate: true,
    silent: true,
  });

  const inpaintMaskAdapters = manager.compositor.getVisibleAdaptersOfType('inpaint_mask');
  const maskImage = await manager.compositor.getCompositeImageDTO(inpaintMaskAdapters, bbox.rect, {
    is_intermediate: true,
    silent: true,
  });

  if (!isEqual(scaledSize, originalSize)) {
    // Scale before processing requires some resizing
    const i2l = g.addNode({
      id: i2lNodeType,
      type: i2lNodeType,
      image: initialImage.image_name ? { image_name: initialImage.image_name } : undefined,
      ...(i2lNodeType === 'i2l' ? { fp32 } : {}),
    });

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
      invert: !canvasSettings.preserveMask,
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
    if (modelLoader.type !== 'flux_model_loader' && modelLoader.type !== 'sd3_model_loader') {
      g.addEdge(modelLoader, 'unet', createGradientMask, 'unet');
    }
    g.addEdge(resizeImageToScaledSize, 'image', createGradientMask, 'image');
    g.addEdge(resizeMaskToScaledSize, 'image', createGradientMask, 'mask');

    g.addEdge(createGradientMask, 'denoise_mask', denoise, 'denoise_mask');

    // After denoising, resize the image and mask back to original size
    g.addEdge(l2i, 'image', resizeImageToOriginalSize, 'image');
    g.addEdge(createGradientMask, 'expanded_mask_area', resizeMaskToOriginalSize, 'image');

    // Finally, paste the generated masked image back onto the original image
    g.addEdge(resizeImageToOriginalSize, 'image', canvasPasteBack, 'generated_image');
    g.addEdge(resizeMaskToOriginalSize, 'image', canvasPasteBack, 'mask');

    // Do the paste back if we are sending to gallery (in which case we want to see the full image), or if we are sending
    // to canvas but not outputting only masked regions
    if (!canvasSettings.sendToCanvas || !canvasSettings.outputOnlyMaskedRegions) {
      canvasPasteBack.source_image = { image_name: initialImage.image_name };
    }

    return canvasPasteBack;
  } else {
    // No scale before processing, much simpler
    const i2l = g.addNode({
      id: i2lNodeType,
      type: i2lNodeType,
      image: initialImage.image_name ? { image_name: initialImage.image_name } : undefined,
      ...(i2lNodeType === 'i2l' ? { fp32 } : {}),
    });

    const alphaToMask = g.addNode({
      id: getPrefixedId('alpha_to_mask'),
      type: 'tomask',
      image: { image_name: maskImage.image_name },
      invert: !canvasSettings.preserveMask,
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
    if (modelLoader.type !== 'flux_model_loader' && modelLoader.type !== 'sd3_model_loader') {
      g.addEdge(modelLoader, 'unet', createGradientMask, 'unet');
    }
    g.addEdge(createGradientMask, 'denoise_mask', denoise, 'denoise_mask');
    g.addEdge(createGradientMask, 'expanded_mask_area', canvasPasteBack, 'mask');

    g.addEdge(l2i, 'image', canvasPasteBack, 'generated_image');

    // Do the paste back if we are sending to gallery (in which case we want to see the full image), or if we are sending
    // to canvas but not outputting only masked regions
    if (!canvasSettings.sendToCanvas || !canvasSettings.outputOnlyMaskedRegions) {
      canvasPasteBack.source_image = { image_name: initialImage.image_name };
    }

    return canvasPasteBack;
  }
};
