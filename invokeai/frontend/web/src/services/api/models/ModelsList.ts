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
import type { LoRAModelConfig } from './LoRAModelConfig';
import type { StableDiffusion1ModelCheckpointConfig } from './StableDiffusion1ModelCheckpointConfig';
import type { StableDiffusion1ModelDiffusersConfig } from './StableDiffusion1ModelDiffusersConfig';
import type { StableDiffusion2ModelCheckpointConfig } from './StableDiffusion2ModelCheckpointConfig';
import type { StableDiffusion2ModelDiffusersConfig } from './StableDiffusion2ModelDiffusersConfig';
import type { TextualInversionModelConfig } from './TextualInversionModelConfig';
import type { VaeModelConfig } from './VaeModelConfig';

export type ModelsList = {
<<<<<<< HEAD
  models: Record<string, Record<string, Record<string, (StableDiffusion2DiffusersModelConfig | ControlNetModelConfig | LoraModelConfig | StableDiffusion1CheckpointModelConfig | TextualInversionModelConfig | StableDiffusion1DiffusersModelConfig | StableDiffusion2CheckpointModelConfig | VAEModelConfig)>>>;
>>>>>>> 76dd749b1 (chore: Rebuild API)
=======
  models: Record<string, Record<string, Record<string, (StableDiffusion1ModelDiffusersConfig | StableDiffusion2ModelCheckpointConfig | TextualInversionModelConfig | ControlNetModelConfig | VaeModelConfig | StableDiffusion2ModelDiffusersConfig | LoRAModelConfig | StableDiffusion1ModelCheckpointConfig)>>>;
>>>>>>> 0f3b7d2b3 (chore: Rebuild API with new Model API names)
};
