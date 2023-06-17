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
  models: Record<string, Record<string, Record<string, (TextualInversionModelConfig | StableDiffusion2ModelDiffusersConfig | ControlNetModelConfig | StableDiffusion2ModelCheckpointConfig | StableDiffusion1ModelCheckpointConfig | VaeModelConfig | StableDiffusion1ModelDiffusersConfig | LoRAModelConfig)>>>;
};
