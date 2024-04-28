import { Button, Flex, Heading, SimpleGrid, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useMainModelDefaultSettings } from 'features/modelManagerV2/hooks/useMainModelDefaultSettings';
import { DefaultHeight } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/DefaultHeight';
import { DefaultWidth } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/DefaultWidth';
import type { ParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';

import { DefaultCfgRescaleMultiplier } from './DefaultCfgRescaleMultiplier';
import { DefaultCfgScale } from './DefaultCfgScale';
import { DefaultScheduler } from './DefaultScheduler';
import { DefaultSteps } from './DefaultSteps';
import { DefaultVae } from './DefaultVae';
import { DefaultVaePrecision } from './DefaultVaePrecision';

export interface FormField<T> {
  value: T;
  isEnabled: boolean;
}

export type MainModelDefaultSettingsFormData = {
  vae: FormField<string>;
  vaePrecision: FormField<'fp16' | 'fp32'>;
  scheduler: FormField<ParameterScheduler>;
  steps: FormField<number>;
  cfgScale: FormField<number>;
  cfgRescaleMultiplier: FormField<number>;
  width: FormField<number>;
  height: FormField<number>;
};

export const MainModelDefaultSettings = () => {
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const {
    defaultSettingsDefaults,
    isLoading: isLoadingDefaultSettings,
    optimalDimension,
  } = useMainModelDefaultSettings(selectedModelKey);

  const [updateModel, { isLoading: isLoadingUpdateModel }] = useUpdateModelMutation();

  const { handleSubmit, control, formState, reset } = useForm<MainModelDefaultSettingsFormData>({
    defaultValues: defaultSettingsDefaults,
  });

  const onSubmit = useCallback<SubmitHandler<MainModelDefaultSettingsFormData>>(
    (data) => {
      if (!selectedModelKey) {
        return;
      }

      const body = {
        vae: data.vae.isEnabled ? data.vae.value : null,
        vae_precision: data.vaePrecision.isEnabled ? data.vaePrecision.value : null,
        cfg_scale: data.cfgScale.isEnabled ? data.cfgScale.value : null,
        cfg_rescale_multiplier: data.cfgRescaleMultiplier.isEnabled ? data.cfgRescaleMultiplier.value : null,
        steps: data.steps.isEnabled ? data.steps.value : null,
        scheduler: data.scheduler.isEnabled ? data.scheduler.value : null,
        width: data.width.isEnabled ? data.width.value : null,
        height: data.height.isEnabled ? data.height.value : null,
      };

      updateModel({
        key: selectedModelKey,
        body: { default_settings: body },
      })
        .unwrap()
        .then((_) => {
          dispatch(
            addToast(
              makeToast({
                title: t('modelManager.defaultSettingsSaved'),
                status: 'success',
              })
            )
          );
          reset(data);
        })
        .catch((error) => {
          if (error) {
            dispatch(
              addToast(
                makeToast({
                  title: `${error.data.detail} `,
                  status: 'error',
                })
              )
            );
          }
        });
    },
    [selectedModelKey, dispatch, reset, updateModel, t]
  );

  if (isLoadingDefaultSettings) {
    return <Text>{t('common.loading')}</Text>;
  }

  return (
    <>
      <Flex gap="4" justifyContent="space-between" w="full" pb={4}>
        <Heading fontSize="md">{t('modelManager.defaultSettings')}</Heading>
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

      <SimpleGrid columns={2} gap={8}>
        <DefaultVae control={control} name="vae" />
        <DefaultVaePrecision control={control} name="vaePrecision" />
        <DefaultScheduler control={control} name="scheduler" />
        <DefaultSteps control={control} name="steps" />
        <DefaultCfgScale control={control} name="cfgScale" />
        <DefaultCfgRescaleMultiplier control={control} name="cfgRescaleMultiplier" />
        <DefaultWidth control={control} optimalDimension={optimalDimension} />
        <DefaultHeight control={control} optimalDimension={optimalDimension} />
      </SimpleGrid>
    </>
  );
};
