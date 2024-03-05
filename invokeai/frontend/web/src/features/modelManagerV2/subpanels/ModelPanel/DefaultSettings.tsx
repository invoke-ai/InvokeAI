import { Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { isNil } from 'lodash-es';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

import { DefaultSettingsForm } from './DefaultSettings/DefaultSettingsForm';

const initialStatesSelector = createMemoizedSelector(selectConfigSlice, (config) => {
  const { steps, guidance, scheduler, cfgRescaleMultiplier, vaePrecision } = config.sd;

  return {
    initialSteps: steps.initial,
    initialCfg: guidance.initial,
    initialScheduler: scheduler,
    initialCfgRescaleMultiplier: cfgRescaleMultiplier.initial,
    initialVaePrecision: vaePrecision,
  };
});

export const DefaultSettings = () => {
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { t } = useTranslation();

  const { data, isLoading } = useGetModelConfigQuery(selectedModelKey ?? skipToken);
  const { initialSteps, initialCfg, initialScheduler, initialCfgRescaleMultiplier, initialVaePrecision } =
    useAppSelector(initialStatesSelector);

  const defaultSettingsDefaults = useMemo(() => {
    return {
      vae: { isEnabled: !isNil(data?.default_settings?.vae), value: data?.default_settings?.vae || 'default' },
      vaePrecision: {
        isEnabled: !isNil(data?.default_settings?.vae_precision),
        value: data?.default_settings?.vae_precision || initialVaePrecision || 'fp32',
      },
      scheduler: {
        isEnabled: !isNil(data?.default_settings?.scheduler),
        value: data?.default_settings?.scheduler || initialScheduler || 'euler',
      },
      steps: { isEnabled: !isNil(data?.default_settings?.steps), value: data?.default_settings?.steps || initialSteps },
      cfgScale: {
        isEnabled: !isNil(data?.default_settings?.cfg_scale),
        value: data?.default_settings?.cfg_scale || initialCfg,
      },
      cfgRescaleMultiplier: {
        isEnabled: !isNil(data?.default_settings?.cfg_rescale_multiplier),
        value: data?.default_settings?.cfg_rescale_multiplier || initialCfgRescaleMultiplier,
      },
    };
  }, [
    data?.default_settings,
    initialSteps,
    initialCfg,
    initialScheduler,
    initialCfgRescaleMultiplier,
    initialVaePrecision,
  ]);

  if (isLoading) {
    return <Text>{t('common.loading')}</Text>;
  }

  return <DefaultSettingsForm defaultSettingsDefaults={defaultSettingsDefaults} />;
};
