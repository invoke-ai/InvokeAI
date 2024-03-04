import { Button, Flex, Heading } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { ParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { IoPencil } from 'react-icons/io5';
import { useUpdateModelMetadataMutation } from 'services/api/endpoints/models';

import { DefaultCfgRescaleMultiplier } from './DefaultCfgRescaleMultiplier';
import { DefaultCfgScale } from './DefaultCfgScale';
import { DefaultScheduler } from './DefaultScheduler';
import { DefaultSteps } from './DefaultSteps';
import { DefaultVae } from './DefaultVae';
import { DefaultVaePrecision } from './DefaultVaePrecision';
import { SettingToggle } from './SettingToggle';

export interface FormField<T> {
  value: T;
  isEnabled: boolean;
}

export type DefaultSettingsFormData = {
  vae: FormField<string>;
  vaePrecision: FormField<string>;
  scheduler: FormField<ParameterScheduler>;
  steps: FormField<number>;
  cfgScale: FormField<number>;
  cfgRescaleMultiplier: FormField<number>;
};

export const DefaultSettingsForm = ({
  defaultSettingsDefaults,
}: {
  defaultSettingsDefaults: DefaultSettingsFormData;
}) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);

  const [editModelMetadata, { isLoading }] = useUpdateModelMetadataMutation();

  const { handleSubmit, control, formState } = useForm<DefaultSettingsFormData>({
    defaultValues: defaultSettingsDefaults,
  });

  const onSubmit = useCallback<SubmitHandler<DefaultSettingsFormData>>(
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
      };

      editModelMetadata({
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
    [selectedModelKey, dispatch, editModelMetadata, t]
  );

  return (
    <>
      <Flex gap="2" justifyContent="space-between" w="full" mb={5}>
        <Heading fontSize="md">{t('modelManager.defaultSettings')}</Heading>
        <Button
          size="sm"
          leftIcon={<IoPencil />}
          colorScheme="invokeYellow"
          isDisabled={!formState.isDirty}
          onClick={handleSubmit(onSubmit)}
          type="submit"
          isLoading={isLoading}
        >
          {t('common.save')}
        </Button>
      </Flex>

      <Flex flexDir="column" gap={8}>
        <Flex gap={8}>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="vae" />
            <DefaultVae control={control} name="vae" />
          </Flex>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="vaePrecision" />
            <DefaultVaePrecision control={control} name="vaePrecision" />
          </Flex>
        </Flex>
        <Flex gap={8}>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="scheduler" />
            <DefaultScheduler control={control} name="scheduler" />
          </Flex>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="steps" />
            <DefaultSteps control={control} name="steps" />
          </Flex>
        </Flex>
        <Flex gap={8}>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="cfgScale" />
            <DefaultCfgScale control={control} name="cfgScale" />
          </Flex>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="cfgRescaleMultiplier" />
            <DefaultCfgRescaleMultiplier control={control} name="cfgRescaleMultiplier" />
          </Flex>
        </Flex>
      </Flex>
    </>
  );
};
