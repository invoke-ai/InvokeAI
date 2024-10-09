import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { isNil } from 'lodash-es';
import { useMemo } from 'react';
import type { MainModelConfig } from 'services/api/types';

const initialStatesSelector = createMemoizedSelector(selectConfigSlice, (config) => {
  const { steps, guidance, scheduler, cfgRescaleMultiplier, vaePrecision, width, height } = config.sd;
  const { guidance: fluxGuidance } = config.flux;

  return {
    initialSteps: steps.initial,
    initialCfg: guidance.initial,
    initialScheduler: scheduler,
    initialCfgRescaleMultiplier: cfgRescaleMultiplier.initial,
    initialVaePrecision: vaePrecision,
    initialWidth: width.initial,
    initialHeight: height.initial,
    initialGuidance: fluxGuidance.initial,
  };
});

export const useMainModelDefaultSettings = (modelConfig: MainModelConfig) => {
  const {
    initialSteps,
    initialCfg,
    initialScheduler,
    initialCfgRescaleMultiplier,
    initialVaePrecision,
    initialWidth,
    initialHeight,
    initialGuidance,
  } = useAppSelector(initialStatesSelector);

  const defaultSettingsDefaults = useMemo(() => {
    return {
      vae: {
        isEnabled: !isNil(modelConfig?.default_settings?.vae),
        value: modelConfig?.default_settings?.vae || 'default',
      },
      vaePrecision: {
        isEnabled: !isNil(modelConfig?.default_settings?.vae_precision),
        value: modelConfig?.default_settings?.vae_precision || initialVaePrecision || 'fp32',
      },
      scheduler: {
        isEnabled: !isNil(modelConfig?.default_settings?.scheduler),
        value: modelConfig?.default_settings?.scheduler || initialScheduler || 'dpmpp_3m_k',
      },
      steps: {
        isEnabled: !isNil(modelConfig?.default_settings?.steps),
        value: modelConfig?.default_settings?.steps || initialSteps,
      },
      cfgScale: {
        isEnabled: !isNil(modelConfig?.default_settings?.cfg_scale),
        value: modelConfig?.default_settings?.cfg_scale || initialCfg,
      },
      cfgRescaleMultiplier: {
        isEnabled: !isNil(modelConfig?.default_settings?.cfg_rescale_multiplier),
        value: modelConfig?.default_settings?.cfg_rescale_multiplier || initialCfgRescaleMultiplier,
      },
      width: {
        isEnabled: !isNil(modelConfig?.default_settings?.width),
        value: modelConfig?.default_settings?.width || initialWidth,
      },
      height: {
        isEnabled: !isNil(modelConfig?.default_settings?.height),
        value: modelConfig?.default_settings?.height || initialHeight,
      },
      guidance: {
        isEnabled: !isNil(modelConfig?.default_settings?.guidance),
        value: modelConfig?.default_settings?.guidance || initialGuidance,
      },
    };
  }, [
    modelConfig,
    initialVaePrecision,
    initialScheduler,
    initialSteps,
    initialCfg,
    initialCfgRescaleMultiplier,
    initialWidth,
    initialHeight,
    initialGuidance,
  ]);

  return defaultSettingsDefaults;
};
