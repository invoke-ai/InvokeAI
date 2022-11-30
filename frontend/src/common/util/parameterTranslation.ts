import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { OptionsState } from 'features/options/store/optionsSlice';
import { SystemState } from 'features/system/store/systemSlice';

import { stringToSeedWeightsArray } from './seedWeightPairs';
import randomInt from './randomInt';
import { InvokeTabName } from 'features/tabs/components/InvokeTabs';
import {
  CanvasState,
  isCanvasMaskLine,
} from 'features/canvas/store/canvasTypes';
import generateMask from 'features/canvas/util/generateMask';
import openBase64ImageInTab from './openBase64ImageInTab';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';

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
  const canvasBaseLayer = getCanvasBaseLayer();

  const {
    generationMode,
    optionsState,
    canvasState,
    systemState,
    imageToProcessUrl,
  } = config;

  const {
    cfgScale,
    codeformerFidelity,
    facetoolStrength,
    facetoolType,
    height,
    hiresFix,
    img2imgStrength,
    infillMethod,
    initialImage,
    iterations,
    perlin,
    prompt,
    sampler,
    seamBlur,
    seamless,
    seamSize,
    seamSteps,
    seamStrength,
    seed,
    seedWeights,
    shouldFitToWidthHeight,
    shouldGenerateVariations,
    shouldRandomizeSeed,
    shouldRunESRGAN,
    shouldRunFacetool,
    steps,
    threshold,
    tileSize,
    upscalingLevel,
    upscalingStrength,
    variationAmount,
    width,
  } = optionsState;

  const {
    shouldDisplayInProgressType,
    saveIntermediatesInterval,
    enableImageDebugging,
  } = systemState;

  const generationParameters: { [k: string]: any } = {
    prompt,
    iterations,
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

  let esrganParameters: false | { [k: string]: any } = false;
  let facetoolParameters: false | { [k: string]: any } = false;

  generationParameters.seed = shouldRandomizeSeed
    ? randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)
    : seed;

  // parameters common to txt2img and img2img
  if (['txt2img', 'img2img'].includes(generationMode)) {
    generationParameters.seamless = seamless;
    generationParameters.hires_fix = hiresFix;

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
  }

  // img2img exclusive parameters
  if (generationMode === 'img2img' && initialImage) {
    generationParameters.init_img =
      typeof initialImage === 'string' ? initialImage : initialImage.url;
    generationParameters.strength = img2imgStrength;
    generationParameters.fit = shouldFitToWidthHeight;
  }

  // inpainting exclusive parameters
  if (generationMode === 'unifiedCanvas' && canvasBaseLayer) {
    const {
      layerState: { objects },
      boundingBoxCoordinates,
      boundingBoxDimensions,
      inpaintReplace,
      shouldUseInpaintReplace,
      stageScale,
      isMaskEnabled,
      shouldPreserveMaskedArea,
      boundingBoxScaleMethod: boundingBoxScale,
      scaledBoundingBoxDimensions,
    } = canvasState;

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

    const tempScale = canvasBaseLayer.scale();

    canvasBaseLayer.scale({
      x: 1 / stageScale,
      y: 1 / stageScale,
    });

    const absPos = canvasBaseLayer.getAbsolutePosition();

    const imageDataURL = canvasBaseLayer.toDataURL({
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

    canvasBaseLayer.scale(tempScale);

    generationParameters.init_img = imageDataURL;

    generationParameters.progress_images = false;

    if (boundingBoxScale !== 'none') {
      generationParameters.inpaint_width = scaledBoundingBoxDimensions.width;
      generationParameters.inpaint_height = scaledBoundingBoxDimensions.height;
    }

    generationParameters.seam_size = seamSize;
    generationParameters.seam_blur = seamBlur;
    generationParameters.seam_strength = seamStrength;
    generationParameters.seam_steps = seamSteps;
    generationParameters.tile_size = tileSize;
    generationParameters.infill_method = infillMethod;
    generationParameters.force_outpaint = false;
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

  if (enableImageDebugging) {
    generationParameters.enable_image_debugging = enableImageDebugging;
  }

  return {
    generationParameters,
    esrganParameters,
    facetoolParameters,
  };
};
