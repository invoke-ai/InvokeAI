import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetRuntimeConfigQuery, useUpdateRuntimeConfigMutation } from 'services/api/endpoints/appInfo';

type ModelCacheSleepTimerOption = {
  label: string;
  value: string;
};

const modelCacheSleepTimerOptions = [
  { label: 'settings.modelCacheSleepTimerOff', value: '0' },
  { label: 'settings.modelCacheSleepTimer1Min', value: '1' },
  { label: 'settings.modelCacheSleepTimer5Min', value: '5' },
  { label: 'settings.modelCacheSleepTimer10Min', value: '10' },
  { label: 'settings.modelCacheSleepTimer30Min', value: '30' },
] satisfies ModelCacheSleepTimerOption[];

const getModelCacheSleepTimerOption = (minutes: number): ModelCacheSleepTimerOption => {
  const value = String(minutes);
  return (
    modelCacheSleepTimerOptions.find((option) => option.value === value) ?? {
      label: 'settings.modelCacheSleepTimerCustom',
      value,
    }
  );
};

export const SettingsModelCacheSleepTimerSelect = memo(() => {
  const { t } = useTranslation();
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: runtimeConfig } = useGetRuntimeConfigQuery();
  const [updateRuntimeConfig, { isLoading }] = useUpdateRuntimeConfigMutation();
  const modelCacheSleepTimer = runtimeConfig?.config.model_cache_keep_alive_min ?? 0;
  const canEditRuntimeConfig = runtimeConfig ? !runtimeConfig.config.multiuser || currentUser?.is_admin : false;

  const options = useMemo(
    () =>
      modelCacheSleepTimerOptions.map((option) => ({
        ...option,
        label: t(option.label),
      })),
    [t]
  );

  const value = useMemo(() => {
    const option = getModelCacheSleepTimerOption(modelCacheSleepTimer);
    return {
      ...option,
      label:
        option.label === 'settings.modelCacheSleepTimerCustom'
          ? t(option.label, { minutes: modelCacheSleepTimer })
          : t(option.label),
    };
  }, [modelCacheSleepTimer, t]);

  const onChange = useCallback<ComboboxOnChange>(
    async (selection) => {
      const minutes = Number(selection?.value);
      if (!Number.isFinite(minutes) || minutes < 0 || minutes === modelCacheSleepTimer) {
        return;
      }

      try {
        await updateRuntimeConfig({ model_cache_keep_alive_min: minutes }).unwrap();
      } catch {
        toast({
          id: 'SETTINGS_MODEL_CACHE_SLEEP_TIMER_SAVE_FAILED',
          title: t('settings.modelCacheSleepTimerSaveFailed'),
          status: 'error',
        });
      }
    },
    [modelCacheSleepTimer, t, updateRuntimeConfig]
  );

  return (
    <FormControl>
      <FormLabel>{t('settings.modelCacheSleepTimer')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        isDisabled={!runtimeConfig || !canEditRuntimeConfig || isLoading}
      />
    </FormControl>
  );
});

SettingsModelCacheSleepTimerSelect.displayName = 'SettingsModelCacheSleepTimerSelect';
