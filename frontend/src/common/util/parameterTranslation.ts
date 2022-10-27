import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from '../../app/constants';
import { OptionsState } from '../../features/options/optionsSlice';
import { SystemState } from '../../features/system/systemSlice';

import { stringToSeedWeightsArray } from './seedWeightPairs';
import randomInt from './randomInt';
import { InvokeTabName } from '../../features/tabs/InvokeTabs';
import { InpaintingState } from '../../features/tabs/Inpainting/inpaintingSlice';
import generateMask from '../../features/tabs/Inpainting/util/generateMask';

export type FrontendToBackendParametersConfig = {
  generationMode: InvokeTabName;
  optionsState: OptionsState;
  inpaintingState: InpaintingState;
  systemState: SystemState;
  imageToProcessUrl?: string;
  maskImageElement?: HTMLImageElement;
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
    inpaintingState,
    systemState,
    imageToProcessUrl,
    maskImageElement,
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
    initialImagePath,
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

  const { shouldDisplayInProgress } = systemState;

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
    progress_images: shouldDisplayInProgress,
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
  if (generationMode === 'img2img') {
    generationParameters.init_img = initialImagePath;
    generationParameters.strength = img2imgStrength;
    generationParameters.fit = shouldFitToWidthHeight;
  }

  // inpainting exclusive parameters
  if (generationMode === 'inpainting' && maskImageElement) {
    const {
      lines,
      boundingBoxCoordinate: { x, y },
      boundingBoxDimensions: { width, height },
      shouldShowBoundingBox,
      inpaintReplace,
      shouldUseInpaintReplace,
    } = inpaintingState;

    let bx = x,
      by = y,
      bwidth = width,
      bheight = height;

    if (!shouldShowBoundingBox) {
      bx = 0;
      by = 0;
      bwidth = maskImageElement.width;
      bheight = maskImageElement.height;
    }

    const boundingBox = {
      x: bx,
      y: by,
      width: bwidth,
      height: bheight,
    };

    if (shouldUseInpaintReplace) {
      generationParameters.inpaint_replace = inpaintReplace;
    }

    generationParameters.init_img = imageToProcessUrl;
    generationParameters.strength = img2imgStrength;
    generationParameters.fit = false;

    const maskDataURL = generateMask(maskImageElement, lines, boundingBox);

    generationParameters.init_mask = maskDataURL.split(
      'data:image/png;base64,'
    )[1];

    generationParameters.bounding_box = boundingBox;
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

  return {
    generationParameters,
    esrganParameters,
    facetoolParameters,
  };
};
