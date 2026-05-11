import { Button, Flex, Heading, SimpleGrid } from '@invoke-ai/ui-library';
import { useIsModelManagerEnabled } from 'features/modelManagerV2/hooks/useIsModelManagerEnabled';
import { useLoRAModelDefaultSettings } from 'features/modelManagerV2/hooks/useLoRAModelDefaultSettings';
import { DefaultWeight } from 'features/modelManagerV2/subpanels/ModelPanel/LoRAModelDefaultSettings/DefaultWeight';
import { DefaultWeightMax } from 'features/modelManagerV2/subpanels/ModelPanel/LoRAModelDefaultSettings/DefaultWeightMax';
import { DefaultWeightMin } from 'features/modelManagerV2/subpanels/ModelPanel/LoRAModelDefaultSettings/DefaultWeightMin';
import type { FormField } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/MainModelDefaultSettings';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useEffect } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';
import type { LoRAModelConfig } from 'services/api/types';

export type LoRAModelDefaultSettingsFormData = {
  weight: FormField<number>;
  weightMin: FormField<number>;
  weightMax: FormField<number>;
};

type Props = {
  modelConfig: LoRAModelConfig;
};

export const LoRAModelDefaultSettings = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const canManageModels = useIsModelManagerEnabled();

  const defaultSettingsDefaults = useLoRAModelDefaultSettings(modelConfig);

  const [updateModel, { isLoading: isLoadingUpdateModel }] = useUpdateModelMutation();

  const { handleSubmit, control, formState, reset, setError } = useForm<LoRAModelDefaultSettingsFormData>({
    defaultValues: defaultSettingsDefaults,
  });

  useEffect(() => {
    reset(defaultSettingsDefaults);
  }, [defaultSettingsDefaults, reset]);

  const onSubmit = useCallback<SubmitHandler<LoRAModelDefaultSettingsFormData>>(
    (data) => {
      const weightMin = data.weightMin.isEnabled ? data.weightMin.value : null;
      const weightMax = data.weightMax.isEnabled ? data.weightMax.value : null;

      if (weightMin !== null && weightMax !== null && weightMin >= weightMax) {
        setError('weightMin', { type: 'manual', message: t('lora.weightMinMustBeLessThanMax') });
        toast({
          id: 'DEFAULT_SETTINGS_SAVE_FAILED',
          title: t('lora.weightMinMustBeLessThanMax'),
          status: 'error',
        });
        return;
      }

      const body = {
        weight: data.weight.isEnabled ? data.weight.value : null,
        weight_min: weightMin,
        weight_max: weightMax,
      };

      updateModel({
        key: modelConfig.key,
        body: { default_settings: body },
      })
        .unwrap()
        .then((_) => {
          toast({
            id: 'DEFAULT_SETTINGS_SAVED',
            title: t('modelManager.defaultSettingsSaved'),
            status: 'success',
          });
          reset(data);
        })
        .catch((error) => {
          if (error) {
            toast({
              id: 'DEFAULT_SETTINGS_SAVE_FAILED',
              title: `${error.data.detail} `,
              status: 'error',
            });
          }
        });
    },
    [updateModel, modelConfig.key, t, reset, setError]
  );

  return (
    <>
      <Flex gap="4" justifyContent="space-between" w="full" pb={4}>
        <Heading fontSize="md">{t('modelManager.defaultSettings')}</Heading>
        {canManageModels && (
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
        )}
      </Flex>

      <SimpleGrid columns={3} gap={8}>
        <DefaultWeight control={control} name="weight" />
        <DefaultWeightMin control={control} name="weightMin" />
        <DefaultWeightMax control={control} name="weightMax" />
      </SimpleGrid>
    </>
  );
});

LoRAModelDefaultSettings.displayName = 'LoRAModelDefaultSettings';
