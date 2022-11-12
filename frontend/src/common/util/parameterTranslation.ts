import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { OptionsState } from 'features/options/optionsSlice';
import { SystemState } from 'features/system/systemSlice';

import { stringToSeedWeightsArray } from './seedWeightPairs';
import randomInt from './randomInt';
import { InvokeTabName } from 'features/tabs/InvokeTabs';
import { CanvasState, isCanvasMaskLine } from 'features/canvas/canvasSlice';
import generateMask from 'features/canvas/util/generateMask';
import { canvasImageLayerRef } from 'features/canvas/IAICanvas';
import openBase64ImageInTab from './openBase64ImageInTab';

export type FrontendToBackendParametersConfig = {
  generationMode: InvokeTabName;
  optionsState: OptionsState;
  canvasState: CanvasState;
  systemState: SystemState;
  imageToProcessUrl?: string;
};

/**
 * Translates/formats frontend state into parameters suitable
 * for consumption by the API.
 */
export const frontendToBackendParameters = (
  config: FrontendToBackendParametersConfig
): { [key: string]: any } => {
  const {
    generationMode,
    optionsState,
    canvasState,
    systemState,
    imageToProcessUrl,
  } = config;

  const {
    prompt,
    iterations,
    steps,
    cfgScale,
    threshold,
    perlin,
    height,
    width,
    sampler,
    seed,
    seamless,
    hiresFix,
    img2imgStrength,
    initialImage,
    shouldFitToWidthHeight,
    shouldGenerateVariations,
    variationAmount,
    seedWeights,
    shouldRunESRGAN,
    upscalingLevel,
    upscalingStrength,
    shouldRunFacetool,
    facetoolStrength,
    codeformerFidelity,
    facetoolType,
    shouldRandomizeSeed,
  } = optionsState;

  const {
    shouldDisplayInProgressType,
    saveIntermediatesInterval,
    enableImageDebugging,
  } = systemState;

  const generationParameters: { [k: string]: any } = {
    prompt,
    iterations:
      shouldRandomizeSeed || shouldGenerateVariations ? iterations : 1,
    steps,
    cfg_scale: cfgScale,
    threshold,
    perlin,
    height,
    width,
    sampler_name: sampler,
    seed,
    progress_images: shouldDisplayInProgressType === 'full-res',
    progress_latents: shouldDisplayInProgressType === 'latents',
    save_intermediates: saveIntermediatesInterval,
    generation_mode: generationMode,
    init_mask: '',
  };

  generationParameters.seed = shouldRandomizeSeed
    ? randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)
    : seed;

  // parameters common to txt2img and img2img
  if (['txt2img', 'img2img'].includes(generationMode)) {
    generationParameters.seamless = seamless;
    generationParameters.hires_fix = hiresFix;
  }

  // img2img exclusive parameters
  if (generationMode === 'img2img' && initialImage) {
    generationParameters.init_img =
      typeof initialImage === 'string' ? initialImage : initialImage.url;
    generationParameters.strength = img2imgStrength;
    generationParameters.fit = shouldFitToWidthHeight;
  }

  // inpainting exclusive parameters
  if (
    ['inpainting', 'outpainting'].includes(generationMode) &&
    canvasImageLayerRef.current
  ) {
    const {
      objects,
      boundingBoxCoordinates,
      boundingBoxDimensions,
      inpaintReplace,
      shouldUseInpaintReplace,
      stageScale,
      isMaskEnabled,
      shouldPreserveMaskedArea,
    } = canvasState[canvasState.currentCanvas];

    const boundingBox = {
      ...boundingBoxCoordinates,
      ...boundingBoxDimensions,
    };

    const maskDataURL = generateMask(
      isMaskEnabled ? objects.filter(isCanvasMaskLine) : [],
      boundingBox
    );

    generationParameters.init_mask = maskDataURL;

    generationParameters.fit = false;

    generationParameters.init_img = imageToProcessUrl;
    generationParameters.strength = img2imgStrength;

    generationParameters.invert_mask = shouldPreserveMaskedArea;

    if (shouldUseInpaintReplace) {
      generationParameters.inpaint_replace = inpaintReplace;
    }

    generationParameters.bounding_box = boundingBox;

    if (generationMode === 'outpainting') {
      const tempScale = canvasImageLayerRef.current.scale();

      canvasImageLayerRef.current.scale({
        x: 1 / stageScale,
        y: 1 / stageScale,
      });

      const absPos = canvasImageLayerRef.current.getAbsolutePosition();

      const imageDataURL = canvasImageLayerRef.current.toDataURL({
        x: boundingBox.x + absPos.x,
        y: boundingBox.y + absPos.y,
        width: boundingBox.width,
        height: boundingBox.height,
      });

      if (enableImageDebugging) {
        openBase64ImageInTab([
          { base64: maskDataURL, caption: 'mask sent as init_mask' },
          { base64: imageDataURL, caption: 'image sent as init_img' },
        ]);
      }

      canvasImageLayerRef.current.scale(tempScale);

      generationParameters.init_img = imageDataURL;

      // TODO: The server metadata generation needs to be changed to fix this.
      generationParameters.progress_images = false;

      generationParameters.seam_size = 96;
      generationParameters.seam_blur = 16;
      generationParameters.seam_strength = 0.7;
      generationParameters.seam_steps = 10;
      generationParameters.tile_size = 32;
      generationParameters.force_outpaint = false;
    }
  }

  if (shouldGenerateVariations) {
    generationParameters.variation_amount = variationAmount;
    if (seedWeights) {
      generationParameters.with_variations =
        stringToSeedWeightsArray(seedWeights);
    }
  } else {
    generationParameters.variation_amount = 0;
  }

  let esrganParameters: false | { [k: string]: any } = false;
  let facetoolParameters: false | { [k: string]: any } = false;

  if (shouldRunESRGAN) {
    esrganParameters = {
      level: upscalingLevel,
      strength: upscalingStrength,
    };
  }

  if (shouldRunFacetool) {
    facetoolParameters = {
      type: facetoolType,
      strength: facetoolStrength,
    };
    if (facetoolType === 'codeformer') {
      facetoolParameters.codeformer_fidelity = codeformerFidelity;
    }
  }

  if (enableImageDebugging) {
    generationParameters.enable_image_debugging = enableImageDebugging;
  }

  return {
    generationParameters,
    esrganParameters,
    facetoolParameters,
  };
};
