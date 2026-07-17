import { CpuOnlyModelSettings } from 'features/modelManagerV2/subpanels/ModelPanel/CpuOnlyModelSettings/CpuOnlyModelSettings';
import { memo } from 'react';
import type { VAEModelConfig } from 'services/api/types';

type Props = {
  modelConfig: VAEModelConfig;
};

export const VAEModelSettings = memo(({ modelConfig }: Props) => {
  return (
    <CpuOnlyModelSettings
      modelConfig={modelConfig}
      feature="cpuOnlyVae"
      label="modelManager.runVaeOnCpu"
      toastIdBase="VAE_SETTINGS"
    />
  );
});

VAEModelSettings.displayName = 'VAEModelSettings';
