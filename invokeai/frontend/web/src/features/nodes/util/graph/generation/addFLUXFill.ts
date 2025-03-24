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

type AddFLUXFillArg = {
  state: RootState;
  g: Graph;
  manager: CanvasManager;
  l2i: Invocation<'flux_vae_decode'>;
  denoise: Invocation<'flux_denoise'>;
  originalSize: Dimensions;
  scaledSize: Dimensions;
};

export const addFLUXFill = async ({
  state,
  g,
  manager,
  l2i,
  denoise,
  originalSize,
  scaledSize,
}: AddFLUXFillArg): Promise<Invocation<'invokeai_img_blend' | 'apply_mask_to_image'>> => {
  // FLUX Fill always fully denoises
  denoise.denoising_start = 0;
  denoise.denoising_end = 1;

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

  const fluxFill = g.addNode({ type: 'flux_fill', id: getPrefixedId('flux_fill') });

  const needsScaleBeforeProcessing = !isEqual(scaledSize, originalSize);

  if (needsScaleBeforeProcessing) {
    // Scale before processing requires some resizing

    // Combine the inpaint mask and the initial image's alpha channel into a single mask
    const maskAlphaToMask = g.addNode({
      id: getPrefixedId('alpha_to_mask'),
      type: 'tomask',
      image: { image_name: maskImage.image_name },
      invert: !canvasSettings.preserveMask,
    });
    const initialImageAlphaToMask = g.addNode({
      id: getPrefixedId('image_alpha_to_mask'),
      type: 'tomask',
      image: { image_name: initialImage.image_name },
    });
    const maskCombine = g.addNode({
      id: getPrefixedId('mask_combine'),
      type: 'mask_combine',
    });
    g.addEdge(maskAlphaToMask, 'image', maskCombine, 'mask1');
    g.addEdge(initialImageAlphaToMask, 'image', maskCombine, 'mask2');

    // Resize the combined and initial image to the scaled size
    const resizeInputMaskToScaledSize = g.addNode({
      id: getPrefixedId('resize_mask_to_scaled_size'),
      type: 'img_resize',
      ...scaledSize,
    });
    g.addEdge(maskCombine, 'image', resizeInputMaskToScaledSize, 'image');

    const alphaMaskToTensorMask = g.addNode({
      type: 'image_mask_to_tensor',
      id: getPrefixedId('image_mask_to_tensor'),
    });
    g.addEdge(resizeInputMaskToScaledSize, 'image', alphaMaskToTensorMask, 'image');
    g.addEdge(alphaMaskToTensorMask, 'mask', fluxFill, 'mask');

    // Resize the initial image to the scaled size and add to the FLUX Fill node
    const resizeInputImageToScaledSize = g.addNode({
      id: getPrefixedId('resize_image_to_scaled_size'),
      type: 'img_resize',
      image: { image_name: initialImage.image_name },
      ...scaledSize,
    });
    g.addEdge(resizeInputImageToScaledSize, 'image', fluxFill, 'image');

    // Provide the FLUX Fill conditioning w/ image and mask to the denoise node
    g.addEdge(fluxFill, 'fill_cond', denoise, 'fill_conditioning');

    // Resize the output image back to the original size
    const resizeOutputImageToOriginalSize = g.addNode({
      id: getPrefixedId('resize_image_to_original_size'),
      type: 'img_resize',
      ...originalSize,
    });
    // After denoising, resize the image and mask back to original size
    g.addEdge(l2i, 'image', resizeOutputImageToOriginalSize, 'image');

    const expandMask = g.addNode({
      type: 'expand_mask_with_fade',
      id: getPrefixedId('expand_mask_with_fade'),
      fade_size_px: params.maskBlur,
    });
    g.addEdge(maskCombine, 'image', expandMask, 'mask');

    // Do the paste back if we are sending to gallery (in which case we want to see the full image), or if we are sending
    // to canvas but not outputting only masked regions
    if (!canvasSettings.sendToCanvas || !canvasSettings.outputOnlyMaskedRegions) {
      const imageLayerBlend = g.addNode({
        type: 'invokeai_img_blend',
        id: getPrefixedId('image_layer_blend'),
        layer_base: { image_name: initialImage.image_name },
      });
      g.addEdge(resizeOutputImageToOriginalSize, 'image', imageLayerBlend, 'layer_upper');
      g.addEdge(expandMask, 'image', imageLayerBlend, 'mask');
      return imageLayerBlend;
    } else {
      // Otherwise, just apply the mask
      const applyMaskToImage = g.addNode({
        type: 'apply_mask_to_image',
        id: getPrefixedId('apply_mask_to_image'),
        invert_mask: true,
      });
      g.addEdge(expandMask, 'image', applyMaskToImage, 'mask');
      g.addEdge(resizeOutputImageToOriginalSize, 'image', applyMaskToImage, 'image');
      return applyMaskToImage;
    }
  } else {
    // Combine the inpaint mask and the initial image's alpha channel into a single mask
    const maskAlphaToMask = g.addNode({
      id: getPrefixedId('alpha_to_mask'),
      type: 'tomask',
      image: { image_name: maskImage.image_name },
      invert: !canvasSettings.preserveMask,
    });
    const initialImageAlphaToMask = g.addNode({
      id: getPrefixedId('image_alpha_to_mask'),
      type: 'tomask',
      image: { image_name: initialImage.image_name },
    });
    const maskCombine = g.addNode({
      id: getPrefixedId('mask_combine'),
      type: 'mask_combine',
    });
    g.addEdge(maskAlphaToMask, 'image', maskCombine, 'mask1');
    g.addEdge(initialImageAlphaToMask, 'image', maskCombine, 'mask2');

    const alphaMaskToTensorMask = g.addNode({
      type: 'image_mask_to_tensor',
      id: getPrefixedId('image_mask_to_tensor'),
    });
    g.addEdge(maskCombine, 'image', alphaMaskToTensorMask, 'image');
    g.addEdge(alphaMaskToTensorMask, 'mask', fluxFill, 'mask');

    fluxFill.image = { image_name: initialImage.image_name };
    g.addEdge(fluxFill, 'fill_cond', denoise, 'fill_conditioning');

    const expandMask = g.addNode({
      type: 'expand_mask_with_fade',
      id: getPrefixedId('expand_mask_with_fade'),
      fade_size_px: params.maskBlur,
    });
    g.addEdge(maskCombine, 'image', expandMask, 'mask');

    // Do the paste back if we are sending to gallery (in which case we want to see the full image), or if we are sending
    // to canvas but not outputting only masked regions
    if (!canvasSettings.sendToCanvas || !canvasSettings.outputOnlyMaskedRegions) {
      const imageLayerBlend = g.addNode({
        type: 'invokeai_img_blend',
        id: getPrefixedId('image_layer_blend'),
        layer_base: { image_name: initialImage.image_name },
      });
      g.addEdge(l2i, 'image', imageLayerBlend, 'layer_upper');
      g.addEdge(expandMask, 'image', imageLayerBlend, 'mask');
      return imageLayerBlend;
    } else {
      // Otherwise, just apply the mask
      const applyMaskToImage = g.addNode({
        type: 'apply_mask_to_image',
        id: getPrefixedId('apply_mask_to_image'),
        invert_mask: true,
      });
      g.addEdge(expandMask, 'image', applyMaskToImage, 'mask');
      g.addEdge(l2i, 'image', applyMaskToImage, 'image');
      return applyMaskToImage;
    }
  }
};
