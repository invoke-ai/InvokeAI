// friendly node ids
export const POSITIVE_CONDITIONING = 'positive_conditioning';
export const NEGATIVE_CONDITIONING = 'negative_conditioning';
export const DENOISE_LATENTS = 'denoise_latents';
export const DENOISE_LATENTS_HRF = 'denoise_latents_hrf';
export const LATENTS_TO_IMAGE = 'latents_to_image';
export const LATENTS_TO_IMAGE_HRF_HR = 'latents_to_image_hrf_hr';
export const LATENTS_TO_IMAGE_HRF_LR = 'latents_to_image_hrf_lr';
export const IMAGE_TO_LATENTS_HRF = 'image_to_latents_hrf';
export const RESIZE_HRF = 'resize_hrf';
export const ESRGAN_HRF = 'esrgan_hrf';
export const NSFW_CHECKER = 'nsfw_checker';
export const WATERMARKER = 'invisible_watermark';
export const NOISE = 'noise';
export const NOISE_HRF = 'noise_hrf';
export const RANDOM_INT = 'rand_int';
export const RANGE_OF_SIZE = 'range_of_size';
export const ITERATE = 'iterate';
export const MAIN_MODEL_LOADER = 'main_model_loader';
export const VAE_LOADER = 'vae_loader';
export const LORA_LOADER = 'lora_loader';
export const CLIP_SKIP = 'clip_skip';
export const IMAGE_TO_LATENTS = 'image_to_latents';
export const LATENTS_TO_LATENTS = 'latents_to_latents';
export const RESIZE = 'resize_image';
export const IMG2IMG_RESIZE = 'img2img_resize';
export const CANVAS_OUTPUT = 'canvas_output';
export const INPAINT_IMAGE = 'inpaint_image';
export const SCALED_INPAINT_IMAGE = 'scaled_inpaint_image';
export const INPAINT_IMAGE_RESIZE_UP = 'inpaint_image_resize_up';
export const INPAINT_IMAGE_RESIZE_DOWN = 'inpaint_image_resize_down';
export const INPAINT_INFILL = 'inpaint_infill';
export const INPAINT_INFILL_RESIZE_DOWN = 'inpaint_infill_resize_down';
export const INPAINT_FINAL_IMAGE = 'inpaint_final_image';
export const INPAINT_CREATE_MASK = 'inpaint_create_mask';
export const INPAINT_MASK = 'inpaint_mask';
export const CANVAS_COHERENCE_DENOISE_LATENTS = 'canvas_coherence_denoise_latents';
export const CANVAS_COHERENCE_NOISE = 'canvas_coherence_noise';
export const CANVAS_COHERENCE_NOISE_INCREMENT = 'canvas_coherence_noise_increment';
export const CANVAS_COHERENCE_MASK_EDGE = 'canvas_coherence_mask_edge';
export const CANVAS_COHERENCE_INPAINT_CREATE_MASK = 'canvas_coherence_inpaint_create_mask';
export const MASK_FROM_ALPHA = 'tomask';
export const MASK_EDGE = 'mask_edge';
export const MASK_BLUR = 'mask_blur';
export const MASK_COMBINE = 'mask_combine';
export const MASK_RESIZE_UP = 'mask_resize_up';
export const MASK_RESIZE_DOWN = 'mask_resize_down';
export const COLOR_CORRECT = 'color_correct';
export const PASTE_IMAGE = 'img_paste';
export const CONTROL_NET_COLLECT = 'control_net_collect';
export const IP_ADAPTER_COLLECT = 'ip_adapter_collect';
export const T2I_ADAPTER_COLLECT = 't2i_adapter_collect';
export const IP_ADAPTER = 'ip_adapter';
export const DYNAMIC_PROMPT = 'dynamic_prompt';
export const IMAGE_COLLECTION = 'image_collection';
export const IMAGE_COLLECTION_ITERATE = 'image_collection_iterate';
export const METADATA = 'core_metadata';
export const BATCH_METADATA = 'batch_metadata';
export const BATCH_METADATA_COLLECT = 'batch_metadata_collect';
export const BATCH_SEED = 'batch_seed';
export const BATCH_PROMPT = 'batch_prompt';
export const BATCH_STYLE_PROMPT = 'batch_style_prompt';
export const METADATA_COLLECT = 'metadata_collect';
export const MERGE_METADATA = 'merge_metadata';
export const ESRGAN = 'esrgan';
export const DIVIDE = 'divide';
export const SCALE = 'scale_image';
export const SDXL_MODEL_LOADER = 'sdxl_model_loader';
export const SDXL_DENOISE_LATENTS = 'sdxl_denoise_latents';
export const SDXL_REFINER_MODEL_LOADER = 'sdxl_refiner_model_loader';
export const SDXL_REFINER_POSITIVE_CONDITIONING = 'sdxl_refiner_positive_conditioning';
export const SDXL_REFINER_NEGATIVE_CONDITIONING = 'sdxl_refiner_negative_conditioning';
export const SDXL_REFINER_DENOISE_LATENTS = 'sdxl_refiner_denoise_latents';
export const SDXL_REFINER_INPAINT_CREATE_MASK = 'refiner_inpaint_create_mask';
export const SEAMLESS = 'seamless';
export const SDXL_REFINER_SEAMLESS = 'refiner_seamless';

// these image-outputting nodes are from the linear UI and we should not handle the gallery logic on them
// instead, we wait for LINEAR_UI_OUTPUT node, and handle it like any other image-outputting node
export const nodeIDDenyList = [
  CANVAS_OUTPUT,
  LATENTS_TO_IMAGE,
  LATENTS_TO_IMAGE_HRF_HR,
  NSFW_CHECKER,
  WATERMARKER,
  ESRGAN,
  ESRGAN_HRF,
  RESIZE_HRF,
  LATENTS_TO_IMAGE_HRF_LR,
  IMG2IMG_RESIZE,
  INPAINT_IMAGE,
  SCALED_INPAINT_IMAGE,
  INPAINT_IMAGE_RESIZE_UP,
  INPAINT_IMAGE_RESIZE_DOWN,
  INPAINT_INFILL,
  INPAINT_INFILL_RESIZE_DOWN,
  INPAINT_FINAL_IMAGE,
  INPAINT_CREATE_MASK,
  INPAINT_MASK,
  PASTE_IMAGE,
  SCALE,
];

// friendly graph ids
export const TEXT_TO_IMAGE_GRAPH = 'text_to_image_graph';
export const IMAGE_TO_IMAGE_GRAPH = 'image_to_image_graph';
export const CANVAS_TEXT_TO_IMAGE_GRAPH = 'canvas_text_to_image_graph';
export const CANVAS_IMAGE_TO_IMAGE_GRAPH = 'canvas_image_to_image_graph';
export const CANVAS_INPAINT_GRAPH = 'canvas_inpaint_graph';
export const CANVAS_OUTPAINT_GRAPH = 'canvas_outpaint_graph';
export const SDXL_TEXT_TO_IMAGE_GRAPH = 'sdxl_text_to_image_graph';
export const SDXL_IMAGE_TO_IMAGE_GRAPH = 'sxdl_image_to_image_graph';
export const SDXL_CANVAS_TEXT_TO_IMAGE_GRAPH = 'sdxl_canvas_text_to_image_graph';
export const SDXL_CANVAS_IMAGE_TO_IMAGE_GRAPH = 'sdxl_canvas_image_to_image_graph';
export const SDXL_CANVAS_INPAINT_GRAPH = 'sdxl_canvas_inpaint_graph';
export const SDXL_CANVAS_OUTPAINT_GRAPH = 'sdxl_canvas_outpaint_graph';
