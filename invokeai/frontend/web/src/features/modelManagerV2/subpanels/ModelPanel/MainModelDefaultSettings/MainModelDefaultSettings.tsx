import { Button, Flex, Heading, SimpleGrid } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useMainModelDefaultSettings } from 'features/modelManagerV2/hooks/useMainModelDefaultSettings';
import { selectSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { DefaultHeight } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/DefaultHeight';
import { DefaultWidth } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/DefaultWidth';
import type { ParameterScheduler } from 'features/parameters/types/parameterSchemas';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';
import type { MainModelConfig } from 'services/api/types';

import { DefaultCfgRescaleMultiplier } from './DefaultCfgRescaleMultiplier';
import { DefaultCfgScale } from './DefaultCfgScale';
import { DefaultGuidance } from './DefaultGuidance';
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
  guidance: FormField<number>;
};

type Props = {
  modelConfig: MainModelConfig;
};

export const MainModelDefaultSettings = memo(({ modelConfig }: Props) => {
  const selectedModelKey = useAppSelector(selectSelectedModelKey);
  const { t } = useTranslation();

  const isFlux = useMemo(() => {
    return modelConfig.base === 'flux';
  }, [modelConfig]);

  const defaultSettingsDefaults = useMainModelDefaultSettings(modelConfig);
  const optimalDimension = useMemo(() => {
    const modelBase = modelConfig?.base;
    return getOptimalDimension(modelBase ?? null);
  }, [modelConfig]);
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
        guidance: data.guidance.isEnabled ? data.guidance.value : null,
      };

      updateModel({
        key: selectedModelKey,
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
    [selectedModelKey, reset, updateModel, t]
  );

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
        {!isFlux && <DefaultVaePrecision control={control} name="vaePrecision" />}
        {!isFlux && <DefaultScheduler control={control} name="scheduler" />}
        <DefaultSteps control={control} name="steps" />
        {isFlux && <DefaultGuidance control={control} name="guidance" />}
        {!isFlux && <DefaultCfgScale control={control} name="cfgScale" />}
        {!isFlux && <DefaultCfgRescaleMultiplier control={control} name="cfgRescaleMultiplier" />}
        <DefaultWidth control={control} optimalDimension={optimalDimension} />
        <DefaultHeight control={control} optimalDimension={optimalDimension} />
      </SimpleGrid>
    </>
  );
});

MainModelDefaultSettings.displayName = 'MainModelDefaultSettings';
