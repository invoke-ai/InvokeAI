import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { OptionsState } from 'features/options/store/optionsSlice';
import { SystemState } from 'features/system/store/systemSlice';
import { Vector2d } from 'konva/lib/types';
import { Dimensions } from 'features/canvas/store/canvasTypes';

import { stringToSeedWeightsArray } from './seedWeightPairs';
import randomInt from './randomInt';
import { InvokeTabName } from 'features/tabs/tabMap';
import {
  CanvasState,
  isCanvasMaskLine,
} from 'features/canvas/store/canvasTypes';
import generateMask from 'features/canvas/util/generateMask';
import openBase64ImageInTab from './openBase64ImageInTab';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import type {
  UpscalingLevel,
  FacetoolType,
} from 'features/options/store/optionsSlice';

export type FrontendToBackendParametersConfig = {
  generationMode: InvokeTabName;
  optionsState: OptionsState;
  canvasState: CanvasState;
  systemState: SystemState;
  imageToProcessUrl?: string;
};

export type BackendGenerationParameters = {
  prompt: string;
  iterations: number;
  steps: number;
  cfg_scale: number;
  threshold: number;
  perlin: number;
  height: number;
  width: number;
  sampler_name: string;
  seed: number;
  progress_images: boolean;
  progress_latents: boolean;
  save_intermediates: number;
  generation_mode: InvokeTabName;
  init_mask: string;
  init_img?: string;
  fit?: boolean;
  seam_size?: number;
  seam_blur?: number;
  seam_strength?: number;
  seam_steps?: number;
  tile_size?: number;
  infill_method?: string;
  force_outpaint?: boolean;
  seamless?: boolean;
  hires_fix?: boolean;
  strength?: number;
  invert_mask?: boolean;
  inpaint_replace?: number;
  bounding_box?: Vector2d & Dimensions;
  inpaint_width?: number;
  inpaint_height?: number;
  with_variations?: Array<Array<number>>;
  variation_amount?: number;
  enable_image_debugging?: boolean;
};

export type BackendEsrGanParameters = {
  level: UpscalingLevel;
  strength: number;
};

export type BackendFacetoolParameters = {
  type: FacetoolType;
  strength: number;
  codeformer_fidelity?: number;
};

export type BackendParameters = {
  generationParameters: BackendGenerationParameters;
  esrganParameters: false | BackendEsrGanParameters;
  facetoolParameters: false | BackendFacetoolParameters;
};

/**
 * Translates/formats frontend state into parameters suitable
 * for consumption by the API.
 */
export const frontendToBackendParameters = (
  config: FrontendToBackendParametersConfig
): BackendParameters => {
  const canvasBaseLayer = getCanvasBaseLayer();

  const { generationMode, optionsState, canvasState, systemState } = config;

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

  const generationParameters: BackendGenerationParameters = {
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

  let esrganParameters: false | BackendEsrGanParameters = false;
  let facetoolParameters: false | BackendFacetoolParameters = false;

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
