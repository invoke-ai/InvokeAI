import { isNil } from 'es-toolkit/compat';
import { DEFAULT_LORA_WEIGHT_CONFIG } from 'features/controlLayers/store/lorasSlice';
import { useMemo } from 'react';
import type { LoRAModelConfig } from 'services/api/types';

export const useLoRAModelDefaultSettings = (modelConfig: LoRAModelConfig) => {
  const defaultSettingsDefaults = useMemo(() => {
    return {
      weight: {
        isEnabled: !isNil(modelConfig?.default_settings?.weight),
        value: modelConfig?.default_settings?.weight ?? DEFAULT_LORA_WEIGHT_CONFIG.initial,
      },
    };
  }, [modelConfig?.default_settings]);

  return defaultSettingsDefaults;
};
