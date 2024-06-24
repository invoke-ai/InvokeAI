import type { RootState } from 'app/store/store';
import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import {
  CLIP_SKIP,
  CONTROL_LAYERS_GRAPH,
  DENOISE_LATENTS,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  NEGATIVE_CONDITIONING,
  NEGATIVE_CONDITIONING_COLLECT,
  NOISE,
  POSITIVE_CONDITIONING,
  POSITIVE_CONDITIONING_COLLECT,
  VAE_LOADER,
} from 'features/nodes/util/graph/constants';
import { addControlAdapters } from 'features/nodes/util/graph/generation/addControlAdapters';
// import { addHRF } from 'features/nodes/util/graph/generation/addHRF';
import { addIPAdapters } from 'features/nodes/util/graph/generation/addIPAdapters';
import { addLoRAs } from 'features/nodes/util/graph/generation/addLoRAs';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addSeamless } from 'features/nodes/util/graph/generation/addSeamless';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import type { GraphType } from 'features/nodes/util/graph/generation/Graph';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getBoardField, getSizes } from 'features/nodes/util/graph/graphBuilderUtils';
import { isEqual, pick } from 'lodash-es';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

import { addRegions } from './addRegions';

export const buildSD1Graph = async (state: RootState, manager: KonvaNodeManager): Promise<GraphType> => {
  const generationMode = manager.util.getGenerationMode();

  const { bbox, params } = state.canvasV2;

  const {
    model,
    cfgScale: cfg_scale,
    cfgRescaleMultiplier: cfg_rescale_multiplier,
    scheduler,
    steps,
    clipSkip: skipped_layers,
    shouldUseCpuNoise,
    vaePrecision,
    seed,
    vae,
    positivePrompt,
    negativePrompt,
  } = params;

  assert(model, 'No model found in state');

  const { originalSize, scaledSize } = getSizes(bbox);

  const g = new Graph(CONTROL_LAYERS_GRAPH);
  const modelLoader = g.addNode({
    type: 'main_model_loader',
    id: MAIN_MODEL_LOADER,
    model,
  });
  const clipSkip = g.addNode({
    type: 'clip_skip',
    id: CLIP_SKIP,
    skipped_layers,
  });
  const posCond = g.addNode({
    type: 'compel',
    id: POSITIVE_CONDITIONING,
    prompt: positivePrompt,
  });
  const posCondCollect = g.addNode({
    type: 'collect',
    id: POSITIVE_CONDITIONING_COLLECT,
  });
  const negCond = g.addNode({
    type: 'compel',
    id: NEGATIVE_CONDITIONING,
    prompt: negativePrompt,
  });
  const negCondCollect = g.addNode({
    type: 'collect',
    id: NEGATIVE_CONDITIONING_COLLECT,
  });
  const noise = g.addNode({
    type: 'noise',
    id: NOISE,
    seed,
    width: scaledSize.width,
    height: scaledSize.height,
    use_cpu: shouldUseCpuNoise,
  });
  const denoise = g.addNode({
    type: 'denoise_latents',
    id: DENOISE_LATENTS,
    cfg_scale,
    cfg_rescale_multiplier,
    scheduler,
    steps,
    denoising_start: 0,
    denoising_end: 1,
  });
  const l2i = g.addNode({
    type: 'l2i',
    id: LATENTS_TO_IMAGE,
    fp32: vaePrecision === 'fp32',
    board: getBoardField(state),
  });
  const vaeLoader =
    vae?.base === model.base
      ? g.addNode({
          type: 'vae_loader',
          id: VAE_LOADER,
          vae_model: vae,
        })
      : null;

  let imageOutput: Invocation<'l2i' | 'img_nsfw' | 'img_watermark' | 'img_resize' | 'canvas_paste_back'> = l2i;

  g.addEdge(modelLoader, 'unet', denoise, 'unet');
  g.addEdge(modelLoader, 'clip', clipSkip, 'clip');
  g.addEdge(clipSkip, 'clip', posCond, 'clip');
  g.addEdge(clipSkip, 'clip', negCond, 'clip');
  g.addEdge(posCond, 'conditioning', posCondCollect, 'item');
  g.addEdge(negCond, 'conditioning', negCondCollect, 'item');
  g.addEdge(posCondCollect, 'collection', denoise, 'positive_conditioning');
  g.addEdge(negCondCollect, 'collection', denoise, 'negative_conditioning');
  g.addEdge(noise, 'noise', denoise, 'noise');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'sd-1' || modelConfig.base === 'sd-2');

  g.upsertMetadata({
    generation_mode: 'txt2img',
    cfg_scale,
    cfg_rescale_multiplier,
    width: scaledSize.width,
    height: scaledSize.height,
    positive_prompt: positivePrompt,
    negative_prompt: negativePrompt,
    model: Graph.getModelMetadataField(modelConfig),
    seed,
    steps,
    rand_device: shouldUseCpuNoise ? 'cpu' : 'cuda',
    scheduler,
    clip_skip: skipped_layers,
    vae: vae ?? undefined,
  });

  const seamless = addSeamless(state, g, denoise, modelLoader, vaeLoader);

  addLoRAs(state, g, denoise, modelLoader, seamless, clipSkip, posCond, negCond);

  // We might get the VAE from the main model, custom VAE, or seamless node.
  const vaeSource = seamless ?? vaeLoader ?? modelLoader;
  g.addEdge(vaeSource, 'vae', l2i, 'vae');

  if (generationMode === 'txt2img') {
    if (!isEqual(scaledSize, originalSize)) {
      // We need to resize the output image back to the original size
      const resizeImageToOriginalSize = g.addNode({
        id: 'resize_image_to_original_size',
        type: 'img_resize',
        ...originalSize,
      });
      g.addEdge(l2i, 'image', resizeImageToOriginalSize, 'image');

      // This is the new output node
      imageOutput = resizeImageToOriginalSize;
    }
  } else if (generationMode === 'img2img') {
    denoise.denoising_start = 1 - params.img2imgStrength;

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
      imageOutput = resizeImageToOriginalSize;
    } else {
      // No need to resize, just denoise
      const i2l = g.addNode({ id: 'i2l', type: 'i2l', image: { image_name: initialImage.image_name } });
      g.addEdge(vaeSource, 'vae', i2l, 'vae');
      g.addEdge(i2l, 'latents', denoise, 'latents');
    }
  } else if (generationMode === 'inpaint') {
    denoise.denoising_start = 1 - params.img2imgStrength;

    const { compositing } = state.canvasV2;

    const cropBbox = pick(bbox, ['x', 'y', 'width', 'height']);
    const initialImage = await manager.util.getImageSourceImage({
      bbox: cropBbox,
      preview: true,
    });
    const maskImage = await manager.util.getInpaintMaskImage({
      bbox: cropBbox,
      preview: true,
    });

    if (!isEqual(scaledSize, originalSize)) {
      // Scale before processing requires some resizing
      const i2l = g.addNode({ id: 'i2l', type: 'i2l' });
      const resizeImageToScaledSize = g.addNode({
        id: 'resize_image_to_scaled_size',
        type: 'img_resize',
        image: { image_name: initialImage.image_name },
        ...scaledSize,
      });
      const alphaToMask = g.addNode({
        id: 'alpha_to_mask',
        type: 'tomask',
        image: { image_name: maskImage.image_name },
        invert: true,
      });
      const resizeMaskToScaledSize = g.addNode({
        id: 'resize_mask_to_scaled_size',
        type: 'img_resize',
        ...scaledSize,
      });
      const resizeImageToOriginalSize = g.addNode({
        id: 'resize_image_to_original_size',
        type: 'img_resize',
        ...originalSize,
      });
      const resizeMaskToOriginalSize = g.addNode({
        id: 'resize_mask_to_original_size',
        type: 'img_resize',
        ...originalSize,
      });
      const createGradientMask = g.addNode({
        id: 'create_gradient_mask',
        type: 'create_gradient_mask',
        coherence_mode: compositing.canvasCoherenceMode,
        minimum_denoise: compositing.canvasCoherenceMinDenoise,
        edge_radius: compositing.canvasCoherenceEdgeSize,
        fp32: vaePrecision === 'fp32',
      });
      const canvasPasteBack = g.addNode({
        id: 'canvas_paste_back',
        type: 'canvas_paste_back',
        board: getBoardField(state),
        mask_blur: compositing.maskBlur,
        source_image: { image_name: initialImage.image_name },
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
      g.addEdge(resizeImageToOriginalSize, 'image', canvasPasteBack, 'target_image');
      g.addEdge(resizeMaskToOriginalSize, 'image', canvasPasteBack, 'mask');

      imageOutput = canvasPasteBack;
    } else {
      // No scale before processing, much simpler
      const i2l = g.addNode({ id: 'i2l', type: 'i2l', image: { image_name: initialImage.image_name } });
      const alphaToMask = g.addNode({
        id: 'alpha_to_mask',
        type: 'tomask',
        image: { image_name: maskImage.image_name },
        invert: true,
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
        board: getBoardField(state),
        mask_blur: compositing.maskBlur,
        source_image: { image_name: initialImage.image_name },
        mask: { image_name: maskImage.image_name },
      });
      g.addEdge(alphaToMask, 'image', createGradientMask, 'mask');
      g.addEdge(i2l, 'latents', denoise, 'latents');
      g.addEdge(vaeSource, 'vae', i2l, 'vae');
      g.addEdge(vaeSource, 'vae', createGradientMask, 'vae');
      g.addEdge(modelLoader, 'unet', createGradientMask, 'unet');
      g.addEdge(createGradientMask, 'denoise_mask', denoise, 'denoise_mask');
      g.addEdge(l2i, 'image', canvasPasteBack, 'target_image');

      imageOutput = canvasPasteBack;
    }
  }

  const _addedCAs = addControlAdapters(state.canvasV2.controlAdapters.entities, g, denoise, modelConfig.base);
  const _addedIPAs = addIPAdapters(state.canvasV2.ipAdapters.entities, g, denoise, modelConfig.base);
  const _addedRegions = await addRegions(
    manager,
    state.canvasV2.regions.entities,
    g,
    state.canvasV2.document,
    state.canvasV2.bbox,
    modelConfig.base,
    denoise,
    posCond,
    negCond,
    posCondCollect,
    negCondCollect
  );

  // const isHRFAllowed = !addedLayers.some((l) => isInitialImageLayer(l) || isRegionalGuidanceLayer(l));
  // if (isHRFAllowed && state.hrf.hrfEnabled) {
  //   imageOutput = addHRF(state, g, denoise, noise, l2i, vaeSource);
  // }

  if (state.system.shouldUseNSFWChecker) {
    imageOutput = addNSFWChecker(g, imageOutput);
  }

  if (state.system.shouldUseWatermarker) {
    imageOutput = addWatermarker(g, imageOutput);
  }

  // This is the terminal node and must always save to gallery.
  imageOutput.is_intermediate = false;
  imageOutput.use_cache = false;

  g.setMetadataReceivingNode(imageOutput);
  return g.getGraph();
};
