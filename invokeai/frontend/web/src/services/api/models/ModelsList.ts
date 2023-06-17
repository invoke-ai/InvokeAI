/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

<<<<<<< HEAD
import type { invokeai__backend__model_management__models__controlnet__ControlNetModel__Config } from './invokeai__backend__model_management__models__controlnet__ControlNetModel__Config';
import type { invokeai__backend__model_management__models__lora__LoRAModel__Config } from './invokeai__backend__model_management__models__lora__LoRAModel__Config';
import type { invokeai__backend__model_management__models__stable_diffusion__StableDiffusion1Model__CheckpointConfig } from './invokeai__backend__model_management__models__stable_diffusion__StableDiffusion1Model__CheckpointConfig';
import type { invokeai__backend__model_management__models__stable_diffusion__StableDiffusion1Model__DiffusersConfig } from './invokeai__backend__model_management__models__stable_diffusion__StableDiffusion1Model__DiffusersConfig';
import type { invokeai__backend__model_management__models__stable_diffusion__StableDiffusion2Model__CheckpointConfig } from './invokeai__backend__model_management__models__stable_diffusion__StableDiffusion2Model__CheckpointConfig';
import type { invokeai__backend__model_management__models__stable_diffusion__StableDiffusion2Model__DiffusersConfig } from './invokeai__backend__model_management__models__stable_diffusion__StableDiffusion2Model__DiffusersConfig';
import type { invokeai__backend__model_management__models__textual_inversion__TextualInversionModel__Config } from './invokeai__backend__model_management__models__textual_inversion__TextualInversionModel__Config';
import type { invokeai__backend__model_management__models__vae__VaeModel__Config } from './invokeai__backend__model_management__models__vae__VaeModel__Config';

export type ModelsList = {
  models: Record<string, Record<string, Record<string, (invokeai__backend__model_management__models__stable_diffusion__StableDiffusion1Model__DiffusersConfig | invokeai__backend__model_management__models__controlnet__ControlNetModel__Config | invokeai__backend__model_management__models__lora__LoRAModel__Config | invokeai__backend__model_management__models__stable_diffusion__StableDiffusion2Model__DiffusersConfig | invokeai__backend__model_management__models__textual_inversion__TextualInversionModel__Config | invokeai__backend__model_management__models__vae__VaeModel__Config | invokeai__backend__model_management__models__stable_diffusion__StableDiffusion2Model__CheckpointConfig | invokeai__backend__model_management__models__stable_diffusion__StableDiffusion1Model__CheckpointConfig)>>>;
=======
import type { ControlNetModelConfig } from './ControlNetModelConfig';
import type { LoraModelConfig } from './LoraModelConfig';
import type { StableDiffusion1CheckpointModelConfig } from './StableDiffusion1CheckpointModelConfig';
import type { StableDiffusion1DiffusersModelConfig } from './StableDiffusion1DiffusersModelConfig';
import type { StableDiffusion2CheckpointModelConfig } from './StableDiffusion2CheckpointModelConfig';
import type { StableDiffusion2DiffusersModelConfig } from './StableDiffusion2DiffusersModelConfig';
import type { TextualInversionModelConfig } from './TextualInversionModelConfig';
import type { VAEModelConfig } from './VAEModelConfig';

export type ModelsList = {
  models: Record<string, Record<string, Record<string, (StableDiffusion2DiffusersModelConfig | ControlNetModelConfig | LoraModelConfig | StableDiffusion1CheckpointModelConfig | TextualInversionModelConfig | StableDiffusion1DiffusersModelConfig | StableDiffusion2CheckpointModelConfig | VAEModelConfig)>>>;
>>>>>>> 76dd749b1 (chore: Rebuild API)
};
