import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetRuntimeConfigQuery, useUpdateRuntimeConfigMutation } from 'services/api/endpoints/appInfo';
import type { S } from 'services/api/types';

type ImageSubfolderStrategy = NonNullable<S['InvokeAIAppConfig']['image_subfolder_strategy']>;

type ImageSubfolderStrategyOption = {
  label: string;
  value: ImageSubfolderStrategy;
};

export const imageSubfolderStrategyOptions = [
  { label: 'settings.imageSubfolderStrategyFlat', value: 'flat' },
  { label: 'settings.imageSubfolderStrategyDate', value: 'date' },
  { label: 'settings.imageSubfolderStrategyType', value: 'type' },
  { label: 'settings.imageSubfolderStrategyHash', value: 'hash' },
] satisfies ImageSubfolderStrategyOption[];

export const isImageSubfolderStrategy = (value: unknown): value is ImageSubfolderStrategy =>
  imageSubfolderStrategyOptions.some((option) => option.value === value);

export const getImageSubfolderStrategyOption = (strategy: ImageSubfolderStrategy) =>
  imageSubfolderStrategyOptions.find((option) => option.value === strategy);

export const SettingsImageSubfolderStrategySelect = memo(() => {
  const { t } = useTranslation();
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: runtimeConfig } = useGetRuntimeConfigQuery();
  const [updateRuntimeConfig, { isLoading }] = useUpdateRuntimeConfigMutation();
  const imageSubfolderStrategy = runtimeConfig?.config.image_subfolder_strategy ?? 'flat';
  const canEditRuntimeConfig = runtimeConfig ? !runtimeConfig.config.multiuser || currentUser?.is_admin : false;

  const options = useMemo(
    () => imageSubfolderStrategyOptions.map((option) => ({ ...option, label: t(option.label) })),
    [t]
  );

  const value = useMemo(
    () => options.find((option) => option.value === imageSubfolderStrategy),
    [imageSubfolderStrategy, options]
  );

  const onChange = useCallback<ComboboxOnChange>(
    async (selection) => {
      if (!isImageSubfolderStrategy(selection?.value) || selection.value === imageSubfolderStrategy) {
        return;
      }

      try {
        await updateRuntimeConfig({ image_subfolder_strategy: selection.value }).unwrap();
      } catch {
        toast({
          id: 'SETTINGS_IMAGE_SUBFOLDER_STRATEGY_SAVE_FAILED',
          title: t('settings.imageSubfolderStrategySaveFailed'),
          status: 'error',
        });
      }
    },
    [imageSubfolderStrategy, t, updateRuntimeConfig]
  );

  return (
    <FormControl>
      <FormLabel>{t('settings.imageSubfolderStrategy')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        isDisabled={!runtimeConfig || !canEditRuntimeConfig || isLoading}
      />
    </FormControl>
  );
});

SettingsImageSubfolderStrategySelect.displayName = 'SettingsImageSubfolderStrategySelect';
