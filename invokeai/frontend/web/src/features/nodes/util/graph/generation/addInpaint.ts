import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { Dimensions } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { isMainModelWithoutUnet } from 'features/nodes/util/graph/graphBuilderUtils';
import type {
  DenoiseLatentsNodes,
  LatentToImageNodes,
  MainModelLoaderNodes,
  VaeSourceNodes,
} from 'features/nodes/util/graph/types';
import { isEqual } from 'lodash-es';
import type { ImageDTO, Invocation } from 'services/api/types';

type AddInpaintArg = {
  state: RootState;
  g: Graph;
  manager: CanvasManager;
  l2i: Invocation<LatentToImageNodes>;
  i2lNodeType: 'i2l' | 'flux_vae_encode' | 'sd3_i2l' | 'cogview4_i2l';
  denoise: Invocation<DenoiseLatentsNodes>;
  vaeSource: Invocation<VaeSourceNodes | MainModelLoaderNodes>;
  modelLoader: Invocation<MainModelLoaderNodes>;
  originalSize: Dimensions;
  scaledSize: Dimensions;
  denoising_start: number;
  fp32: boolean;
  seed: number;
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
  seed,
}: AddInpaintArg): Promise<Invocation<'invokeai_img_blend' | 'apply_mask_to_image'>> => {
  denoise.denoising_start = denoising_start;

  const params = selectParamsSlice(state);
  const canvasSettings = selectCanvasSettingsSlice(state);
  const canvas = selectCanvasSlice(state);

  const { rect } = canvas.bbox;

  const rasterAdapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
  const initialImage = await manager.compositor.getCompositeImageDTO(rasterAdapters, rect, {
    is_intermediate: true,
    silent: true,
  });

  const inpaintMaskAdapters = manager.compositor.getVisibleAdaptersOfType('inpaint_mask');

  // Get inpaint mask adapters that have noise settings
  const noiseMaskAdapters = inpaintMaskAdapters.filter((adapter) => adapter.state.noiseLevel !== undefined);

  // Create a composite noise mask if we have any adapters with noise settings
  let noiseMaskImage: ImageDTO | null = null;
  if (noiseMaskAdapters.length > 0) {
    noiseMaskImage = await manager.compositor.getGrayscaleMaskCompositeImageDTO(
      noiseMaskAdapters,
      rect,
      'noiseLevel',
      canvasSettings.preserveMask,
      {
        is_intermediate: true,
        silent: true,
      }
    );
  }

  // Create a composite denoise limit mask
  const maskImage = await manager.compositor.getGrayscaleMaskCompositeImageDTO(
    inpaintMaskAdapters, // denoise limit defaults to 1 for masks that don't have it
    rect,
    'denoiseLimit',
    canvasSettings.preserveMask,
    {
      is_intermediate: true,
      silent: true,
    }
  );

  const needsScaleBeforeProcessing = !isEqual(scaledSize, originalSize);

  if (needsScaleBeforeProcessing) {
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

    // If we have a noise mask, apply it to the input image before i2l conversion
    if (noiseMaskImage) {
      // Resize the noise mask to match the scaled size
      const resizeNoiseMaskToScaledSize = g.addNode({
        id: getPrefixedId('resize_noise_mask_to_scaled_size'),
        type: 'img_resize',
        image: { image_name: noiseMaskImage.image_name },
        ...scaledSize,
      });

      // Add noise to the scaled image using the mask
      const noiseNode = g.addNode({
        type: 'img_noise',
        id: getPrefixedId('add_inpaint_noise'),
        noise_type: 'gaussian',
        amount: 1.0, // the mask controls the actual intensity
        noise_color: true,
        seed: seed,
      });

      g.addEdge(resizeImageToScaledSize, 'image', noiseNode, 'image');
      g.addEdge(resizeNoiseMaskToScaledSize, 'image', noiseNode, 'mask');
      g.addEdge(noiseNode, 'image', i2l, 'image');
    } else {
      g.addEdge(resizeImageToScaledSize, 'image', i2l, 'image');
    }

    const resizeMaskToScaledSize = g.addNode({
      id: getPrefixedId('resize_mask_to_scaled_size'),
      type: 'img_resize',
      image: { image_name: maskImage.image_name },
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
    const expandMask = g.addNode({
      type: 'expand_mask_with_fade',
      id: getPrefixedId('expand_mask_with_fade'),
      fade_size_px: params.maskBlur,
    });

    g.addEdge(i2l, 'latents', denoise, 'latents');
    g.addEdge(vaeSource, 'vae', i2l, 'vae');
    g.addEdge(vaeSource, 'vae', createGradientMask, 'vae');
    if (!isMainModelWithoutUnet(modelLoader)) {
      g.addEdge(modelLoader, 'unet', createGradientMask, 'unet');
    }
    g.addEdge(resizeImageToScaledSize, 'image', createGradientMask, 'image');
    g.addEdge(resizeMaskToScaledSize, 'image', createGradientMask, 'mask');

    g.addEdge(createGradientMask, 'denoise_mask', denoise, 'denoise_mask');

    // After denoising, resize the image and mask back to original size
    g.addEdge(l2i, 'image', resizeImageToOriginalSize, 'image');
    g.addEdge(createGradientMask, 'expanded_mask_area', expandMask, 'mask');
    g.addEdge(expandMask, 'image', resizeMaskToOriginalSize, 'image');

    // After denoising, resize the image and mask back to original size
    // Do the paste back if we are not outputting only masked regions
    if (!canvasSettings.outputOnlyMaskedRegions) {
      const imageLayerBlend = g.addNode({
        type: 'invokeai_img_blend',
        id: getPrefixedId('image_layer_blend'),
        layer_base: { image_name: initialImage.image_name },
      });
      g.addEdge(resizeImageToOriginalSize, 'image', imageLayerBlend, 'layer_upper');
      g.addEdge(resizeMaskToOriginalSize, 'image', imageLayerBlend, 'mask');
      return imageLayerBlend;
    } else {
      // Otherwise, just apply the mask
      const applyMaskToImage = g.addNode({
        type: 'apply_mask_to_image',
        id: getPrefixedId('apply_mask_to_image'),
        invert_mask: true,
      });
      g.addEdge(resizeMaskToOriginalSize, 'image', applyMaskToImage, 'mask');
      g.addEdge(resizeImageToOriginalSize, 'image', applyMaskToImage, 'image');
      return applyMaskToImage;
    }
  } else {
    // No scale before processing, much simpler
    const i2l = g.addNode({
      id: i2lNodeType,
      type: i2lNodeType,
      image: initialImage.image_name ? { image_name: initialImage.image_name } : undefined,
      ...(i2lNodeType === 'i2l' ? { fp32 } : {}),
    });

    // If we have a noise mask, apply it to the input image before i2l conversion
    if (noiseMaskImage) {
      // Add noise to the scaled image using the mask
      const noiseNode = g.addNode({
        type: 'img_noise',
        id: getPrefixedId('add_inpaint_noise'),
        image: initialImage.image_name ? { image_name: initialImage.image_name } : undefined,
        noise_type: 'gaussian',
        amount: 1.0, // the mask controls the actual intensity
        noise_color: true,
        seed: seed,
        mask: { image_name: noiseMaskImage.image_name },
      });

      g.addEdge(noiseNode, 'image', i2l, 'image');
    }

    const createGradientMask = g.addNode({
      id: getPrefixedId('create_gradient_mask'),
      type: 'create_gradient_mask',
      coherence_mode: params.canvasCoherenceMode,
      minimum_denoise: params.canvasCoherenceMinDenoise,
      edge_radius: params.canvasCoherenceEdgeSize,
      fp32,
      image: { image_name: initialImage.image_name },
      mask: { image_name: maskImage.image_name },
    });

    g.addEdge(i2l, 'latents', denoise, 'latents');
    g.addEdge(vaeSource, 'vae', i2l, 'vae');
    g.addEdge(vaeSource, 'vae', createGradientMask, 'vae');
    if (!isMainModelWithoutUnet(modelLoader)) {
      g.addEdge(modelLoader, 'unet', createGradientMask, 'unet');
    }
    g.addEdge(createGradientMask, 'denoise_mask', denoise, 'denoise_mask');

    const expandMask = g.addNode({
      type: 'expand_mask_with_fade',
      id: getPrefixedId('expand_mask_with_fade'),
      fade_size_px: params.maskBlur,
    });
    g.addEdge(createGradientMask, 'expanded_mask_area', expandMask, 'mask');

    // Do the paste back if we are not outputting only masked regions
    if (!canvasSettings.outputOnlyMaskedRegions) {
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
