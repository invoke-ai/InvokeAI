import { Button, Flex, FormControl, FormLabel, Heading, Switch } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useVAEModelSettings } from 'features/modelManagerV2/hooks/useVAEModelSettings';
import { selectSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import type { EncoderModelSettingsFormData } from 'features/modelManagerV2/subpanels/ModelPanel/EncoderModelSettings/EncoderModelSettings';
import type { FormField } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/MainModelDefaultSettings';
import { toast } from 'features/toast/toast';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect, useMemo } from 'react';
import type { Control, SubmitHandler } from 'react-hook-form';
import { useController, useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';
import type { VAEModelConfig } from 'services/api/types';

type Props = {
  modelConfig: VAEModelConfig;
};

const DefaultCpuOnly = memo((props: { name: 'cpuOnly'; control: Control<EncoderModelSettingsFormData> }) => {
  const { field } = useController(props);
  const { t } = useTranslation();

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const updatedValue = {
        ...(field.value as FormField<boolean>),
        value: e.target.checked,
        isEnabled: e.target.checked,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return (field.value as FormField<boolean>).value;
  }, [field.value]);

  return (
    <FormControl>
      <InformationalPopover feature="cpuOnly">
        <FormLabel>{t('modelManager.runOnCpu')}</FormLabel>
      </InformationalPopover>
      <Switch isChecked={value} onChange={onChange} />
    </FormControl>
  );
});

DefaultCpuOnly.displayName = 'DefaultCpuOnly';

export const VAEModelSettings = memo(({ modelConfig }: Props) => {
  const selectedModelKey = useAppSelector(selectSelectedModelKey);
  const { t } = useTranslation();

  const settingsDefaults = useVAEModelSettings(modelConfig);
  const [updateModel, { isLoading: isLoadingUpdateModel }] = useUpdateModelMutation();

  const { handleSubmit, control, formState, reset } = useForm<EncoderModelSettingsFormData>({
    defaultValues: settingsDefaults,
  });

  useEffect(() => {
    reset(settingsDefaults);
  }, [settingsDefaults, reset]);

  const onSubmit = useCallback<SubmitHandler<EncoderModelSettingsFormData>>(
    (data) => {
      if (!selectedModelKey) {
        return;
      }

      const body = {
        cpu_only: data.cpuOnly.isEnabled ? data.cpuOnly.value : null,
      };

      updateModel({
        key: selectedModelKey,
        body,
      })
        .unwrap()
        .then((_) => {
          toast({
            id: 'VAE_SETTINGS_SAVED',
            title: t('modelManager.settingsSaved'),
            status: 'success',
          });
          reset(data);
        })
        .catch((error) => {
          if (error) {
            toast({
              id: 'VAE_SETTINGS_SAVE_FAILED',
              title: `${error.data.detail} `,
              status: 'error',
            });
          }
        });
    },
    [selectedModelKey, reset, updateModel, t]
  );

  return (
    <>
      <Flex gap="4" justifyContent="space-between" w="full" pb={4}>
        <Heading fontSize="md">{t('modelManager.settings')}</Heading>
        <Button
          size="sm"
          leftIcon={<PiCheckBold />}
          colorScheme="invokeYellow"
          isDisabled={!formState.isDirty}
          onClick={handleSubmit(onSubmit)}
          isLoading={isLoadingUpdateModel}
        >
          {t('common.save')}
        </Button>
      </Flex>

      <DefaultCpuOnly control={control} name="cpuOnly" />
    </>
  );
});

VAEModelSettings.displayName = 'VAEModelSettings';
