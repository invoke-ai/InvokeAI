/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ControlNetModelConfig } from './ControlNetModelConfig';
import type { LoRAModelConfig } from './LoRAModelConfig';
import type { StableDiffusion1ModelCheckpointConfig } from './StableDiffusion1ModelCheckpointConfig';
import type { StableDiffusion1ModelDiffusersConfig } from './StableDiffusion1ModelDiffusersConfig';
import type { StableDiffusion2ModelCheckpointConfig } from './StableDiffusion2ModelCheckpointConfig';
import type { StableDiffusion2ModelDiffusersConfig } from './StableDiffusion2ModelDiffusersConfig';
import type { TextualInversionModelConfig } from './TextualInversionModelConfig';
import type { VaeModelConfig } from './VaeModelConfig';

export type ModelsList = {
  models: Record<string, Record<string, Record<string, (StableDiffusion1ModelDiffusersConfig | StableDiffusion2ModelCheckpointConfig | TextualInversionModelConfig | ControlNetModelConfig | VaeModelConfig | StableDiffusion2ModelDiffusersConfig | LoRAModelConfig | StableDiffusion1ModelCheckpointConfig)>>>;
};
