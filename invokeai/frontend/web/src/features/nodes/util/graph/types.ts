export type ImageOutputNodes =
  | 'l2i'
  | 'img_nsfw'
  | 'img_watermark'
  | 'img_resize'
  | 'invokeai_img_blend'
  | 'apply_mask_to_image'
  | 'flux_vae_decode'
  | 'sd3_l2i'
  | 'cogview4_l2i';

export type LatentToImageNodes = 'l2i' | 'flux_vae_decode' | 'sd3_l2i' | 'cogview4_l2i';

export type ImageToLatentsNodes = 'i2l' | 'flux_vae_encode' | 'sd3_i2l' | 'cogview4_i2l';

export type DenoiseLatentsNodes = 'denoise_latents' | 'flux_denoise' | 'sd3_denoise' | 'cogview4_denoise';

export type MainModelLoaderNodes =
  | 'main_model_loader'
  | 'sdxl_model_loader'
  | 'flux_model_loader'
  | 'sd3_model_loader'
  | 'cogview4_model_loader';

export type VaeSourceNodes = 'seamless' | 'vae_loader';
export type NoiseNodes = 'noise' | 'flux_denoise' | 'sd3_denoise' | 'cogview4_denoise';

export type ConditioningNodes =
  | 'compel'
  | 'sdxl_compel_prompt'
  | 'flux_text_encoder'
  | 'sd3_text_encoder'
  | 'cogview4_text_encoder';
