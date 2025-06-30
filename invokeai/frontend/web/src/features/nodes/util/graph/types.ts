import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { GenerationMode } from 'features/controlLayers/store/types';
import type { FieldIdentifier } from 'features/nodes/types/field';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';

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

export type GraphBuilderArg =
  | {
      generationMode: Extract<GenerationMode, 'txt2img'>;
      state: RootState;
    }
  | {
      generationMode: Exclude<GenerationMode, 'txt2img'>;
      state: RootState;
      canvasManager: CanvasManager;
    };

export type GraphBuilderReturn = {
  g: Graph;
  seedFieldIdentifier?: FieldIdentifier;
  positivePromptFieldIdentifier: FieldIdentifier;
};

export class UnsupportedGenerationModeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}
