import { isNil } from 'lodash-es';
import { useMemo } from 'react';
import type { ControlLoRAModelConfig, ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';

export const useControlAdapterModelDefaultSettings = (
  modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | ControlLoRAModelConfig
) => {
  const defaultSettingsDefaults = useMemo(() => {
    return {
      preprocessor: {
        isEnabled: !isNil(modelConfig?.default_settings?.preprocessor),
        value: modelConfig?.default_settings?.preprocessor || 'none',
      },
    };
  }, [modelConfig?.default_settings]);

  return defaultSettingsDefaults;
};
