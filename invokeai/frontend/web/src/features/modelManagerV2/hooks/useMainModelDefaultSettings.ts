import { isNil } from 'es-toolkit/compat';
import { useMemo } from 'react';
import type { MainModelConfig } from 'services/api/types';

export const useMainModelDefaultSettings = (modelConfig: MainModelConfig) => {
  const defaultSettingsDefaults = useMemo(() => {
    return {
      vae: {
        isEnabled: !isNil(modelConfig?.default_settings?.vae),
        value: modelConfig?.default_settings?.vae ?? 'default',
      },
      vaePrecision: {
        isEnabled: !isNil(modelConfig?.default_settings?.vae_precision),
        value: modelConfig?.default_settings?.vae_precision ?? 'fp32',
      },
      scheduler: {
        isEnabled: !isNil(modelConfig?.default_settings?.scheduler),
        value: modelConfig?.default_settings?.scheduler ?? 'dpmpp_3m_k',
      },
      steps: {
        isEnabled: !isNil(modelConfig?.default_settings?.steps),
        value: modelConfig?.default_settings?.steps ?? 30,
      },
      cfgScale: {
        isEnabled: !isNil(modelConfig?.default_settings?.cfg_scale),
        value: modelConfig?.default_settings?.cfg_scale ?? 7,
      },
      cfgRescaleMultiplier: {
        isEnabled: !isNil(modelConfig?.default_settings?.cfg_rescale_multiplier),
        value: modelConfig?.default_settings?.cfg_rescale_multiplier ?? 0,
      },
      width: {
        isEnabled: !isNil(modelConfig?.default_settings?.width),
        value: modelConfig?.default_settings?.width ?? 512,
      },
      height: {
        isEnabled: !isNil(modelConfig?.default_settings?.height),
        value: modelConfig?.default_settings?.height ?? 512,
      },
      guidance: {
        isEnabled: !isNil(modelConfig?.default_settings?.guidance),
        value: modelConfig?.default_settings?.guidance ?? 4,
      },
    };
  }, [modelConfig]);

  return defaultSettingsDefaults;
};
