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

  const { shouldDisplayInProgressType, saveIntermediatesInterval } =
    systemState;

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
  if (generationMode === 'inpainting' && maskImageElement) {
    const {
      lines,
      boundingBoxCoordinate,
      boundingBoxDimensions,
      inpaintReplace,
      shouldUseInpaintReplace,
    } = inpaintingState;

    const boundingBox = {
      ...boundingBoxCoordinate,
      ...boundingBoxDimensions,
    };

    generationParameters.init_img = imageToProcessUrl;
    generationParameters.strength = img2imgStrength;
    generationParameters.fit = false;

    const { maskDataURL, isMaskEmpty } = generateMask(
      maskImageElement,
      lines,
      boundingBox
    );

    generationParameters.is_mask_empty = isMaskEmpty;

    generationParameters.init_mask = maskDataURL.split(
      'data:image/png;base64,'
    )[1];

    if (shouldUseInpaintReplace) {
      generationParameters.inpaint_replace = inpaintReplace;
    }

    generationParameters.bounding_box = boundingBox;

    // TODO: The server metadata generation needs to be changed to fix this.
    generationParameters.progress_images = false;
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
