import { CpuOnlyModelSettings } from 'features/modelManagerV2/subpanels/ModelPanel/CpuOnlyModelSettings/CpuOnlyModelSettings';
import { memo } from 'react';
import type {
  CLIPEmbedModelConfig,
  CLIPVisionModelConfig,
  LlavaOnevisionModelConfig,
  Qwen3EncoderModelConfig,
  SigLIPModelConfig,
  T5EncoderModelConfig,
  TextLLMModelConfig,
  WanT5EncoderModelConfig,
} from 'services/api/types';

type EncoderModelConfig =
  | CLIPEmbedModelConfig
  | T5EncoderModelConfig
  | Qwen3EncoderModelConfig
  | CLIPVisionModelConfig
  | SigLIPModelConfig
  | LlavaOnevisionModelConfig
  | TextLLMModelConfig
  | WanT5EncoderModelConfig;

type Props = {
  modelConfig: EncoderModelConfig;
};

export const EncoderModelSettings = memo(({ modelConfig }: Props) => {
  return (
    <CpuOnlyModelSettings
      modelConfig={modelConfig}
      feature="cpuOnly"
      label="modelManager.runOnCpu"
      toastIdBase="ENCODER_SETTINGS"
    />
  );
});

EncoderModelSettings.displayName = 'EncoderModelSettings';
