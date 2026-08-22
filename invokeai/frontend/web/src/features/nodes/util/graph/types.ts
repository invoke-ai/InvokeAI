import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { GenerationMode } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation } from 'services/api/types';

export type ImageOutputNodes =
  | 'l2i'
  | 'img_nsfw'
  | 'img_watermark'
  | 'img_resize'
  | 'invokeai_img_blend'
  | 'apply_mask_to_image'
  | 'flux_pid_decode'
  | 'flux2_pid_decode'
  | 'sd3_pid_decode'
  | 'sdxl_pid_decode'
  | 'z_image_pid_decode'
  | 'qwen_image_pid_decode'
  | 'flux_vae_decode'
  | 'flux2_vae_decode'
  | 'sd3_l2i'
  | 'cogview4_l2i'
  | 'qwen_image_l2i'
  | 'z_image_l2i'
  | 'ernie_image_vae_decode'
  | 'ideogram4_l2i'
  | 'anima_l2i'
  | 'wan_l2i'
  | 'minimax_h3_latents_to_image';

export type LatentToImageNodes =
  | 'l2i'
  | 'flux_vae_decode'
  | 'flux2_vae_decode'
  | 'sd3_l2i'
  | 'cogview4_l2i'
  | 'qwen_image_l2i'
  | 'z_image_l2i'
  | 'ernie_image_vae_decode'
  | 'anima_l2i'
  | 'wan_l2i'
  | 'minimax_h3_latents_to_image';

export type ImageToLatentsNodes =
  | 'i2l'
  | 'flux_vae_encode'
  | 'flux2_vae_encode'
  | 'sd3_i2l'
  | 'cogview4_i2l'
  | 'qwen_image_i2l'
  | 'z_image_i2l'
  | 'anima_i2l'
  | 'wan_i2l';

export type DenoiseLatentsNodes =
  | 'denoise_latents'
  | 'flux_denoise'
  | 'flux2_denoise'
  | 'sd3_denoise'
  | 'cogview4_denoise'
  | 'qwen_image_denoise'
  | 'z_image_denoise'
  | 'ernie_image_denoise'
  | 'krea2_denoise'
  | 'anima_denoise'
  | 'wan_denoise'
  | 'minimax_h3_denoise';

/**
 * Denoise nodes that support masked denoising (inpaint/outpaint). ERNIE-Image's denoise node
 * has no `denoise_mask` input (it is text-to-image only), and MiniMax H3's denoise node has
 * neither `denoise_mask` nor `denoising_start/end` (txt2img/t2v only), so they are excluded.
 */
export type MaskableDenoiseNodes = Exclude<DenoiseLatentsNodes, 'ernie_image_denoise' | 'minimax_h3_denoise'>;

export type MainModelLoaderNodes =
  | 'main_model_loader'
  | 'sdxl_model_loader'
  | 'flux_model_loader'
  | 'flux2_klein_model_loader'
  | 'flux2_dev_model_loader'
  | 'sd3_model_loader'
  | 'cogview4_model_loader'
  | 'qwen_image_model_loader'
  | 'z_image_model_loader'
  | 'ernie_image_model_loader'
  | 'krea2_model_loader'
  | 'anima_model_loader'
  | 'wan_model_loader'
  | 'minimax_h3_model_loader';

export type VaeSourceNodes = 'seamless' | 'vae_loader';

export type GraphBuilderArg = {
  generationMode: GenerationMode;
  state: RootState;
  manager: CanvasManager | null;
};

export type GraphBuilderReturn = {
  g: Graph;
  seed?: Invocation<'integer'>;
  positivePrompt: Invocation<'string'>;
  negativePrompt?: Invocation<'string'>;
};

export class UnsupportedGenerationModeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}
