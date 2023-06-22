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
  models: Array<(StableDiffusion1ModelCheckpointConfig | StableDiffusion1ModelDiffusersConfig | VaeModelConfig | LoRAModelConfig | ControlNetModelConfig | TextualInversionModelConfig | StableDiffusion2ModelCheckpointConfig | StableDiffusion2ModelDiffusersConfig)>;
};

