/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

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
};
