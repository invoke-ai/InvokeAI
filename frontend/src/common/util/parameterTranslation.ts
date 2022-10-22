/*
    These functions translate frontend state into parameters
    suitable for consumption by the backend, and vice-versa.
*/

import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from '../../app/constants';
import { OptionsState } from '../../features/options/optionsSlice';
import { SystemState } from '../../features/system/systemSlice';

import {
  seedWeightsToString,
  stringToSeedWeightsArray,
} from './seedWeightPairs';
import randomInt from './randomInt';
import { tabMap, tab_dict } from '../../features/tabs/InvokeTabs';

export const frontendToBackendParameters = (
  optionsState: OptionsState,
  systemState: SystemState,
  mode: keyof typeof tab_dict,
  inpaintingMask?: string,
  currentImageUrl?: string
): { [key: string]: any } => {
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
    shouldUseInitImage,
    img2imgStrength,
    initialImagePath,
    maskPath,
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
    seamless,
    hires_fix: hiresFix,
    progress_images: shouldDisplayInProgress,
  };

  generationParameters.seed = shouldRandomizeSeed
    ? randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)
    : seed;

  if (mode === 'img2img') {
    generationParameters.init_img = initialImagePath;
    generationParameters.strength = img2imgStrength;
    generationParameters.fit = shouldFitToWidthHeight;
  }

  if (mode === 'inpainting') {
    generationParameters.init_img = currentImageUrl;
    generationParameters.strength = img2imgStrength;
    generationParameters.fit = shouldFitToWidthHeight;
    generationParameters.init_mask = inpaintingMask;
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
      facetoolParameters.codeformer_fidelity = codeformerFidelity
    }
  }

  return {
    generationParameters,
    esrganParameters,
    facetoolParameters,
  };
};
