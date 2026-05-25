import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { DetailerDenoiseConfig } from 'features/controlLayers/store/detailerRuntimeConfig';
import {
  getGroundedSamDetailerRuntimeConfig,
  getGroundedSamDetectorPromptConfig,
} from 'features/controlLayers/store/detailerRuntimeConfig';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { ImageOutputNodes } from 'features/nodes/util/graph/types';
import type { Invocation } from 'services/api/types';
import { assert } from 'tsafe';

type SupportedModelLoader = Invocation<'main_model_loader'> | Invocation<'sdxl_model_loader'>;
type SupportedUnetSource =
  | SupportedModelLoader
  | Invocation<'seamless'>
  | Invocation<'lora_collection_loader'>
  | Invocation<'sdxl_lora_collection_loader'>;
type SupportedVaeSource =
  | Invocation<'main_model_loader'>
  | Invocation<'sdxl_model_loader'>
  | Invocation<'seamless'>
  | Invocation<'vae_loader'>;
type DetailerCropSource = Invocation<'face_off'> | Invocation<'detailer_crop_from_mask'>;
type DetailerDebugImageSource = Invocation<'l2i'> | Invocation<'color_correct'>;

type AddFaceDetailerPassArg = {
  g: Graph;
  state: RootState;
  image: Invocation<ImageOutputNodes>;
  vaeSource: SupportedVaeSource;
  baseDenoise: Invocation<'denoise_latents'>;
  posCondCollect: Invocation<'collect'>;
  negCondCollect: Invocation<'collect'>;
  seed: Invocation<'integer'>;
  fp32: boolean;
  colorCompensation?: 'None' | 'SDXL';
  isDetailerDebugOutputAllowed?: boolean;
  isLegacyDetailerDetectorAllowed?: boolean;
};

const isSupportedModelBase = (base: string | undefined) => base === 'sd-1' || base === 'sd-2' || base === 'sdxl';

const getDenoiseUnetSource = (g: Graph, denoise: Invocation<'denoise_latents'>): SupportedUnetSource => {
  const edge = g.getEdgesTo(denoise, ['unet'])[0];
  assert(edge, 'Denoise node has no UNet source');
  return g.getNode(edge.source.node_id) as SupportedUnetSource;
};

const addSharedDetailDenoisePass = ({
  g,
  params,
  crop,
  denoiseConfig,
  vaeSource,
  unetSource,
  posCondCollect,
  negCondCollect,
  seed,
  fp32,
  colorCompensation = 'None',
}: {
  g: Graph;
  params: ReturnType<typeof selectParamsSlice>;
  crop: DetailerCropSource;
  denoiseConfig: DetailerDenoiseConfig;
  vaeSource: SupportedVaeSource;
  unetSource: SupportedUnetSource;
  posCondCollect: Invocation<'collect'>;
  negCondCollect: Invocation<'collect'>;
  seed: Invocation<'integer'>;
  fp32: boolean;
  colorCompensation?: 'None' | 'SDXL';
}) => {
  const i2l = g.addNode({
    id: getPrefixedId('detailer_i2l'),
    type: 'i2l',
    fp32,
    color_compensation: colorCompensation,
  });
  const noise = g.addNode({
    id: getPrefixedId('detailer_noise'),
    type: 'noise',
    use_cpu: params.shouldUseCpuNoise,
  });
  const createGradientMask = g.addNode({
    id: getPrefixedId('detailer_create_gradient_mask'),
    type: 'create_gradient_mask',
    coherence_mode: params.canvasCoherenceMode,
    minimum_denoise: 0,
    edge_radius: denoiseConfig.edgeRadius,
    fp32,
  });
  const denoise = g.addNode({
    id: getPrefixedId('detailer_denoise_latents'),
    type: 'denoise_latents',
    cfg_scale: denoiseConfig.cfgScale,
    cfg_rescale_multiplier: params.cfgRescaleMultiplier,
    scheduler: params.scheduler,
    steps: denoiseConfig.steps,
    denoising_start: 1 - denoiseConfig.strength,
    denoising_end: 1,
  });
  const l2i = g.addNode({
    id: getPrefixedId('detailer_l2i'),
    type: 'l2i',
    fp32,
  });

  g.addEdge(crop, 'image', i2l, 'image');
  g.addEdge(vaeSource, 'vae', i2l, 'vae');
  g.addEdge(i2l, 'latents', denoise, 'latents');

  if (crop.type === 'face_off') {
    g.addEdge(crop, 'width', noise, 'width');
    g.addEdge(crop, 'height', noise, 'height');
    g.addEdge(crop, 'mask', createGradientMask, 'mask');
  } else {
    g.addEdge(crop, 'processed_width', noise, 'width');
    g.addEdge(crop, 'processed_height', noise, 'height');
    g.addEdge(crop, 'denoise_mask', createGradientMask, 'mask');
  }

  g.addEdge(seed, 'value', noise, 'seed');
  g.addEdge(noise, 'noise', denoise, 'noise');

  g.addEdge(unetSource, 'unet', denoise, 'unet');
  g.addEdge(posCondCollect, 'collection', denoise, 'positive_conditioning');
  g.addEdge(negCondCollect, 'collection', denoise, 'negative_conditioning');

  g.addEdge(crop, 'image', createGradientMask, 'image');
  g.addEdge(vaeSource, 'vae', createGradientMask, 'vae');
  g.addEdge(unetSource, 'unet', createGradientMask, 'unet');
  g.addEdge(createGradientMask, 'denoise_mask', denoise, 'denoise_mask');

  g.addEdge(denoise, 'latents', l2i, 'latents');
  g.addEdge(vaeSource, 'vae', l2i, 'vae');

  return l2i;
};

const addDetailerColorCorrection = ({
  g,
  params,
  crop,
  image,
}: {
  g: Graph;
  params: ReturnType<typeof selectParamsSlice>;
  crop: Invocation<'detailer_crop_from_mask'>;
  image: Invocation<'l2i'>;
}): Invocation<'l2i'> | Invocation<'color_correct'> => {
  if (params.detailerColorCorrectMode === 'off') {
    return image;
  }

  const colorCorrect = g.addNode({
    id: getPrefixedId('detailer_color_correct'),
    type: 'color_correct',
    colorspace: params.detailerColorCorrectMode,
  });

  g.addEdge(image, 'image', colorCorrect, 'base_image');
  g.addEdge(crop, 'image', colorCorrect, 'color_reference');
  // The detailer denoise mask is black = edit, white = preserve. color_correct uses
  // white = original, black = result, so this limits color matching to the edited target.
  g.addEdge(crop, 'denoise_mask', colorCorrect, 'mask');

  return colorCorrect;
};

const addDetailerDebugOutput = ({
  g,
  targetPrompt,
  detectorPrompt,
  samModel,
  image,
  selectBoundingBox,
  tensorMaskToImage,
  crop,
  detailImage,
  paste,
}: {
  g: Graph;
  targetPrompt: string;
  detectorPrompt: string;
  samModel: string;
  image: Invocation<ImageOutputNodes>;
  selectBoundingBox: Invocation<'select_bounding_box'>;
  tensorMaskToImage: Invocation<'tensor_mask_to_image'>;
  crop: Invocation<'detailer_crop_from_mask'>;
  detailImage: DetailerDebugImageSource;
  paste: Invocation<'detailer_paste_crop'>;
}) => {
  const debugCollage = g.addNode({
    id: getPrefixedId('detailer_debug_collage'),
    type: 'detailer_debug_collage',
    target_prompt: targetPrompt,
    detector_prompt: detectorPrompt,
    sam_model: samModel,
  });
  const debugCanvasOutput = g.addNode({
    id: getPrefixedId('detailer_debug_canvas_output'),
    type: 'canvas_output',
    is_intermediate: false,
    use_cache: false,
  });

  g.addEdge(image, 'image', debugCollage, 'base_image');
  g.addEdge(selectBoundingBox, 'collection', debugCollage, 'selected_bounding_boxes');
  g.addEdge(tensorMaskToImage, 'image', debugCollage, 'mask');
  g.addEdge(crop, 'image', debugCollage, 'processed_crop');
  g.addEdge(crop, 'denoise_mask', debugCollage, 'denoise_mask');
  g.addEdge(crop, 'paste_alpha_mask', debugCollage, 'paste_alpha_mask');
  g.addEdge(detailImage, 'image', debugCollage, 'detailed_crop');
  g.addEdge(paste, 'image', debugCollage, 'final_image');
  g.addEdge(crop, 'x', debugCollage, 'x');
  g.addEdge(crop, 'y', debugCollage, 'y');
  g.addEdge(crop, 'original_width', debugCollage, 'original_width');
  g.addEdge(crop, 'original_height', debugCollage, 'original_height');
  g.addEdge(crop, 'processed_width', debugCollage, 'processed_width');
  g.addEdge(crop, 'processed_height', debugCollage, 'processed_height');
  g.addEdge(crop, 'detected', debugCollage, 'detected');
  g.addEdge(debugCollage, 'image', debugCanvasOutput, 'image');
};

const addMediaPipeFaceDetailerPass = ({
  g,
  params,
  image,
  vaeSource,
  unetSource,
  posCondCollect,
  negCondCollect,
  seed,
  fp32,
  colorCompensation,
}: Omit<AddFaceDetailerPassArg, 'state' | 'baseDenoise'> & {
  params: ReturnType<typeof selectParamsSlice>;
  unetSource: SupportedUnetSource;
}): Invocation<ImageOutputNodes> => {
  const faceOff = g.addNode({
    id: getPrefixedId('detailer_face_off'),
    type: 'face_off',
    face_id: params.detailerFaceId,
    minimum_confidence: params.detailerMinConfidence,
    padding: params.detailerPadding,
    x_offset: 0,
    y_offset: 0,
    chunk: false,
  });

  const l2i = addSharedDetailDenoisePass({
    g,
    params,
    crop: faceOff,
    denoiseConfig: {
      cfgScale: params.detailerCfgScale,
      steps: params.detailerSteps,
      strength: params.detailerStrength,
      edgeRadius: params.detailerMaskBlur,
    },
    vaeSource,
    unetSource,
    posCondCollect,
    negCondCollect,
    seed,
    fp32,
    colorCompensation,
  });

  const paste = g.addNode({
    id: getPrefixedId('detailer_img_paste'),
    type: 'img_paste',
    crop: true,
  });

  g.addEdge(image, 'image', faceOff, 'image');
  g.addEdge(image, 'image', paste, 'base_image');
  g.addEdge(l2i, 'image', paste, 'image');
  g.addEdge(faceOff, 'mask', paste, 'mask');
  g.addEdge(faceOff, 'x', paste, 'x');
  g.addEdge(faceOff, 'y', paste, 'y');

  g.upsertMetadata({
    detailer_enabled: true,
    detailer_version: 'v1',
    detailer_type: 'face',
    detailer_detector: 'mediapipe',
    detailer_face_id: params.detailerFaceId,
    detailer_min_confidence: params.detailerMinConfidence,
    detailer_padding: params.detailerPadding,
    detailer_strength: params.detailerStrength,
    detailer_steps: params.detailerSteps,
    detailer_cfg_scale: params.detailerCfgScale,
    detailer_mask_blur: params.detailerMaskBlur,
  });

  return paste;
};

const addGroundedSamFaceDetailerPass = ({
  g,
  params,
  image,
  vaeSource,
  unetSource,
  posCondCollect,
  negCondCollect,
  seed,
  fp32,
  colorCompensation,
  isDetailerDebugOutputAllowed,
}: Omit<AddFaceDetailerPassArg, 'state' | 'baseDenoise'> & {
  params: ReturnType<typeof selectParamsSlice>;
  unetSource: SupportedUnetSource;
}): Invocation<ImageOutputNodes> => {
  const targetPrompt = params.detailerTargetPrompt.trim() || 'face';
  const detectorPromptConfig = getGroundedSamDetectorPromptConfig(targetPrompt);
  const runtimeConfig = getGroundedSamDetailerRuntimeConfig(params, targetPrompt);
  const groundingDino = g.addNode({
    id: getPrefixedId('detailer_grounding_dino'),
    type: 'grounding_dino',
    model: params.detailerDinoModel,
    prompt: detectorPromptConfig.detectorPrompt,
    detection_threshold: params.detailerDetectionThreshold,
  });
  const selectBoundingBox = g.addNode({
    id: getPrefixedId('detailer_select_bounding_box'),
    type: 'select_bounding_box',
    selection_mode: params.detailerFaceSelection,
    index: params.detailerFaceId,
    label_priority: detectorPromptConfig.labelPriority,
  });
  const segmentAnything = g.addNode({
    id: getPrefixedId('detailer_segment_anything'),
    type: 'segment_anything',
    model: params.detailerSamModel,
    apply_polygon_refinement: true,
    mask_filter: 'all',
  });
  const tensorMaskToImage = g.addNode({
    id: getPrefixedId('detailer_tensor_mask_to_image'),
    type: 'tensor_mask_to_image',
  });
  const crop = g.addNode({
    id: getPrefixedId('detailer_crop_from_mask'),
    type: 'detailer_crop_from_mask',
    padding: params.detailerCropPadding,
    mask_expand: params.detailerMaskExpand,
    mask_feather: params.detailerMaskFeather,
    denoise_mask_expand: runtimeConfig.denoiseMaskExpand,
    denoise_mask_feather: runtimeConfig.denoiseMaskFeather,
    paste_mask_expand: runtimeConfig.pasteMaskExpand,
    paste_mask_feather: runtimeConfig.pasteMaskFeather,
    target_size: runtimeConfig.targetSize,
    max_upscale: runtimeConfig.maxUpscale,
    max_process_size: runtimeConfig.maxProcessSize,
    prevent_downscale: runtimeConfig.preventDownscale,
    denoise_mask_contract: runtimeConfig.denoiseMaskContract,
    paste_mask_contract: runtimeConfig.pasteMaskContract,
  });

  g.addEdge(image, 'image', groundingDino, 'image');
  g.addEdge(groundingDino, 'collection', selectBoundingBox, 'collection');
  g.addEdge(selectBoundingBox, 'collection', segmentAnything, 'bounding_boxes');
  g.addEdge(image, 'image', segmentAnything, 'image');
  g.addEdge(segmentAnything, 'mask', tensorMaskToImage, 'mask');
  g.addEdge(image, 'image', crop, 'image');
  g.addEdge(tensorMaskToImage, 'image', crop, 'mask');

  const l2i = addSharedDetailDenoisePass({
    g,
    params,
    crop,
    denoiseConfig: runtimeConfig,
    vaeSource,
    unetSource,
    posCondCollect,
    negCondCollect,
    seed,
    fp32,
    colorCompensation,
  });
  const detailImage = addDetailerColorCorrection({
    g,
    params,
    crop,
    image: l2i,
  });

  const paste = g.addNode({
    id: getPrefixedId('detailer_paste_crop'),
    type: 'detailer_paste_crop',
  });

  g.addEdge(image, 'image', paste, 'base_image');
  g.addEdge(detailImage, 'image', paste, 'image');
  g.addEdge(crop, 'paste_alpha_mask', paste, 'paste_alpha_mask');
  g.addEdge(crop, 'x', paste, 'x');
  g.addEdge(crop, 'y', paste, 'y');
  g.addEdge(crop, 'original_width', paste, 'original_width');
  g.addEdge(crop, 'original_height', paste, 'original_height');

  if (isDetailerDebugOutputAllowed && params.detailerDebugEnabled) {
    addDetailerDebugOutput({
      g,
      targetPrompt,
      detectorPrompt: detectorPromptConfig.detectorPrompt,
      samModel: params.detailerSamModel,
      image,
      selectBoundingBox,
      tensorMaskToImage,
      crop,
      detailImage,
      paste,
    });
  }

  g.upsertMetadata({
    detailer_enabled: true,
    detailer_version: 'v5',
    detailer_type: 'target',
    detailer_detector: params.detailerDetector,
    detailer_quality: params.detailerQuality,
    detailer_target_profile: runtimeConfig.targetProfile,
    detailer_target_prompt: targetPrompt,
    detailer_detector_prompt: detectorPromptConfig.detectorPrompt,
    detailer_upscale_method: params.detailerUpscaleMethod,
    detailer_face_selection: params.detailerFaceSelection,
    detailer_face_id: params.detailerFaceId,
    detailer_dino_model: params.detailerDinoModel,
    detailer_sam_model: params.detailerSamModel,
    detailer_detection_threshold: params.detailerDetectionThreshold,
    detailer_target_size: runtimeConfig.targetSize,
    detailer_param_target_size: params.detailerTargetSize,
    detailer_max_upscale: runtimeConfig.maxUpscale,
    detailer_param_max_upscale: params.detailerMaxUpscale,
    detailer_max_process_size: runtimeConfig.maxProcessSize,
    detailer_param_max_process_size: params.detailerMaxProcessSize,
    detailer_crop_padding: params.detailerCropPadding,
    detailer_mask_expand: params.detailerMaskExpand,
    detailer_mask_feather: params.detailerMaskFeather,
    detailer_denoise_mask_expand: runtimeConfig.denoiseMaskExpand,
    detailer_param_denoise_mask_expand: params.detailerDenoiseMaskExpand,
    detailer_denoise_mask_feather: runtimeConfig.denoiseMaskFeather,
    detailer_param_denoise_mask_feather: params.detailerDenoiseMaskFeather,
    detailer_paste_mask_expand: runtimeConfig.pasteMaskExpand,
    detailer_param_paste_mask_expand: params.detailerPasteMaskExpand,
    detailer_paste_mask_feather: runtimeConfig.pasteMaskFeather,
    detailer_param_paste_mask_feather: params.detailerPasteMaskFeather,
    detailer_denoise_mask_contract: runtimeConfig.denoiseMaskContract,
    detailer_paste_mask_contract: runtimeConfig.pasteMaskContract,
    detailer_prevent_downscale: runtimeConfig.preventDownscale,
    detailer_color_correct_mode: params.detailerColorCorrectMode,
    detailer_strength: runtimeConfig.strength,
    detailer_param_strength: params.detailerStrength,
    detailer_steps: runtimeConfig.steps,
    detailer_param_steps: params.detailerSteps,
    detailer_cfg_scale: runtimeConfig.cfgScale,
    detailer_param_cfg_scale: params.detailerCfgScale,
  });

  return paste;
};

export const addFaceDetailerPass = ({
  g,
  state,
  image,
  vaeSource,
  baseDenoise,
  posCondCollect,
  negCondCollect,
  seed,
  fp32,
  colorCompensation,
  isDetailerDebugOutputAllowed = import.meta.env.MODE === 'development',
  isLegacyDetailerDetectorAllowed = import.meta.env.MODE === 'development',
}: AddFaceDetailerPassArg): Invocation<ImageOutputNodes> => {
  const params = selectParamsSlice(state);

  if (!params.detailerEnabled || !isSupportedModelBase(params.model?.base)) {
    return image;
  }

  const unetSource = getDenoiseUnetSource(g, baseDenoise);

  if (params.detailerDetector === 'mediapipe' && isLegacyDetailerDetectorAllowed) {
    return addMediaPipeFaceDetailerPass({
      g,
      params,
      image,
      vaeSource,
      unetSource,
      posCondCollect,
      negCondCollect,
      seed,
      fp32,
      colorCompensation,
    });
  }

  const effectiveParams =
    params.detailerDetector === 'mediapipe'
      ? ({
          ...params,
          detailerDetector: 'grounding-dino-sam',
        } satisfies typeof params)
      : params;

  return addGroundedSamFaceDetailerPass({
    g,
    params: effectiveParams,
    image,
    vaeSource,
    unetSource,
    posCondCollect,
    negCondCollect,
    seed,
    fp32,
    colorCompensation,
    isDetailerDebugOutputAllowed,
  });
};
