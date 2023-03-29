import { RootState } from 'app/store';
import {
  ImageToImageInvocation,
  RestoreFaceInvocation,
  TextToImageInvocation,
  UpscaleInvocation,
} from 'services/api';

// fe todo fix model type (frontend uses null, backend uses undefined)
// fe todo update front end to store to have whole image field (vs just name)
// be todo add symmetry fields
// be todo variations....

export function buildTxt2ImgNode(
  state: RootState
): Omit<TextToImageInvocation, 'id'> {
  const { generation, system } = state;

  const { shouldDisplayInProgressType, model } = system;

  const {
    prompt,
    seed,
    steps,
    width,
    height,
    cfgScale: cfg_scale,
    sampler,
    seamless,
  } = generation;

  // missing fields in TextToImageInvocation: strength, hires_fix
  return {
    type: 'txt2img',
    prompt,
    seed,
    steps,
    width,
    height,
    cfg_scale,
    sampler_name: sampler as TextToImageInvocation['sampler_name'],
    seamless,
    model,
    progress_images: shouldDisplayInProgressType === 'full-res',
  };
}

export function buildImg2ImgNode(
  state: RootState
): Omit<ImageToImageInvocation, 'id'> {
  const { generation, system } = state;

  const { shouldDisplayInProgressType, openModel: model } = system;

  const {
    prompt,
    seed,
    steps,
    width,
    height,
    cfgScale,
    sampler,
    seamless,
    img2imgStrength: strength,
    shouldFitToWidthHeight: fit,
    initialImage,
  } = generation;

  return {
    type: 'img2img',
    prompt,
    seed,
    steps,
    width,
    height,
    cfg_scale: cfgScale,
    sampler_name: sampler as ImageToImageInvocation['sampler_name'],
    seamless,
    model: model as string | undefined,
    progress_images: shouldDisplayInProgressType === 'full-res',
    image: {
      image_name:
        typeof initialImage === 'string' ? initialImage : initialImage?.url,
    },
    strength,
    fit,
  };
}

export function buildFacetoolNode(
  state: RootState
): Omit<RestoreFaceInvocation, 'id'> {
  const { generation, postprocessing } = state;

  const { initialImage } = generation;

  const { facetoolStrength: strength } = postprocessing;

  // missing fields in RestoreFaceInvocation: type, codeformer_fidelity
  return {
    type: 'restore_face',
    image: {
      image_name:
        typeof initialImage === 'string' ? initialImage : initialImage?.url,
    },
    strength,
  };
}

// is this ESRGAN??
export function buildUpscaleNode(
  state: RootState
): Omit<UpscaleInvocation, 'id'> {
  const { generation, postprocessing } = state;

  const { initialImage } = generation;

  const { upscalingLevel: level, upscalingStrength: strength } = postprocessing;

  // missing fields in UpscaleInvocation: denoise_str
  return {
    type: 'upscale',
    image: {
      image_name:
        typeof initialImage === 'string' ? initialImage : initialImage?.url,
    },
    strength,
    level,
  };
}
