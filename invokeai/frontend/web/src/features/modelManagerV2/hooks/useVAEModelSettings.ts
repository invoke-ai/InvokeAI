import type { EncoderModelSettingsFormData } from 'features/modelManagerV2/subpanels/ModelPanel/EncoderModelSettings/EncoderModelSettings';
import { useMemo } from 'react';
import type { VAEModelConfig } from 'services/api/types';

export const useVAEModelSettings = (modelConfig: VAEModelConfig) => {
  const vaeModelSettingsDefaults = useMemo<EncoderModelSettingsFormData>(() => {
    const cpuOnly = modelConfig.cpu_only ?? false;

    return {
      cpuOnly: {
        value: cpuOnly,
        isEnabled: cpuOnly,
      },
    };
  }, [modelConfig]);

  return vaeModelSettingsDefaults;
};
