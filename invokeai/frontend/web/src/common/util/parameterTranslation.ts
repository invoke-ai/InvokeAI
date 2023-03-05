import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import { Dimensions } from 'features/canvas/store/canvasTypes';
import { GenerationState } from 'features/parameters/store/generationSlice';
import { SystemState } from 'features/system/store/systemSlice';
import { Vector2d } from 'konva/lib/types';

import {
  CanvasState,
  isCanvasMaskLine,
} from 'features/canvas/store/canvasTypes';
import generateMask from 'features/canvas/util/generateMask';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import type {
  FacetoolType,
  UpscalingLevel,
} from 'features/parameters/store/postprocessingSlice';
import { PostprocessingState } from 'features/parameters/store/postprocessingSlice';
import { InvokeTabName } from 'features/ui/store/tabMap';
import openBase64ImageInTab from './openBase64ImageInTab';
import randomInt from './randomInt';
import { stringToSeedWeightsArray } from './seedWeightPairs';

export type FrontendToBackendParametersConfig = {
  generationMode: InvokeTabName;
  generationState: GenerationState;
  postprocessingState: PostprocessingState;
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
  h_symmetry_time_pct?: number;
  v_symmetry_time_pct?: number;
};

export type BackendEsrGanParameters = {
  level: UpscalingLevel;
  denoise_str: number;
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

  const {
    generationMode,
    generationState,
    postprocessingState,
    canvasState,
    systemState,
  } = config;

  const {
    codeformerFidelity,
    facetoolStrength,
    facetoolType,
    hiresFix,
    hiresStrength,
    shouldRunESRGAN,
    shouldRunFacetool,
    upscalingLevel,
    upscalingStrength,
    upscalingDenoising,
  } = postprocessingState;

  const {
    cfgScale,
    height,
    img2imgStrength,
    infillMethod,
    initialImage,
    iterations,
    perlin,
    prompt,
    negativePrompt,
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
    steps,
    threshold,
    tileSize,
    variationAmount,
    width,
    shouldUseSymmetry,
    horizontalSymmetrySteps,
    verticalSymmetrySteps,
  } = generationState;

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

  if (negativePrompt !== '') {
    generationParameters.prompt = `${prompt} [${negativePrompt}]`;
  }

  generationParameters.seed = shouldRandomizeSeed
    ? randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)
    : seed;

  // Symmetry Settings
  if (shouldUseSymmetry) {
    if (horizontalSymmetrySteps > 0) {
      generationParameters.h_symmetry_time_pct = Math.max(
        0,
        Math.min(1, horizontalSymmetrySteps / steps)
      );
    }

    if (verticalSymmetrySteps > 0) {
      generationParameters.v_symmetry_time_pct = Math.max(
        0,
        Math.min(1, verticalSymmetrySteps / steps)
      );
    }
  }

  // txt2img exclusive parameters
  if (generationMode === 'txt2img') {
    generationParameters.hires_fix = hiresFix;

    if (hiresFix) generationParameters.strength = hiresStrength;
  }

  // parameters common to txt2img and img2img
  if (['txt2img', 'img2img'].includes(generationMode)) {
    generationParameters.seamless = seamless;

    if (shouldRunESRGAN) {
      esrganParameters = {
        level: upscalingLevel,
        denoise_str: upscalingDenoising,
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
