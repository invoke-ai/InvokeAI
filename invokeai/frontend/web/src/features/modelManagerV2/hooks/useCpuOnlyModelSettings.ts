import type { FormField } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/MainModelDefaultSettings';
import { useMemo } from 'react';

export type CpuOnlyModelSettingsFormData = {
  cpuOnly: FormField<boolean>;
};

/**
 * Computes the default form state for the "run on CPU only" toggle. Shared by every model type that
 * exposes a standalone `cpu_only` config field (text encoders and VAEs).
 */
export const useCpuOnlyModelSettings = (modelConfig: { cpu_only?: boolean | null }) => {
  return useMemo<CpuOnlyModelSettingsFormData>(() => {
    const cpuOnly = modelConfig.cpu_only ?? false;

    return {
      cpuOnly: {
        value: cpuOnly,
        isEnabled: cpuOnly,
      },
    };
  }, [modelConfig]);
};
