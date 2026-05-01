import type { EncoderModelSettingsFormData } from 'features/modelManagerV2/subpanels/ModelPanel/EncoderModelSettings/EncoderModelSettings';
import { useMemo } from 'react';
import type {
  CLIPEmbedModelConfig,
  CLIPVisionModelConfig,
  LlavaOnevisionModelConfig,
  Qwen3EncoderModelConfig,
  SigLIPModelConfig,
  T5EncoderModelConfig,
  TextLLMModelConfig,
} from 'services/api/types';

type EncoderModelConfig =
  | CLIPEmbedModelConfig
  | T5EncoderModelConfig
  | Qwen3EncoderModelConfig
  | CLIPVisionModelConfig
  | SigLIPModelConfig
  | LlavaOnevisionModelConfig
  | TextLLMModelConfig;

export const useEncoderModelSettings = (modelConfig: EncoderModelConfig) => {
  const encoderModelSettingsDefaults = useMemo<EncoderModelSettingsFormData>(() => {
    const cpuOnly = modelConfig.cpu_only ?? false;

    return {
      cpuOnly: {
        value: cpuOnly,
        isEnabled: cpuOnly,
      },
    };
  }, [modelConfig]);

  return encoderModelSettingsDefaults;
};
