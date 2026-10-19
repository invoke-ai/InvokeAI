import { Button, Flex, Heading, SimpleGrid } from '@invoke-ai/ui-library';
import { useControlAdapterModelDefaultSettings } from 'features/modelManagerV2/hooks/useControlAdapterModelDefaultSettings';
import { useIsModelManagerEnabled } from 'features/modelManagerV2/hooks/useIsModelManagerEnabled';
import { DefaultFp8StorageControlAdapter } from 'features/modelManagerV2/subpanels/ModelPanel/ControlAdapterModelDefaultSettings/DefaultFp8StorageControlAdapter';
import { DefaultPreprocessor } from 'features/modelManagerV2/subpanels/ModelPanel/ControlAdapterModelDefaultSettings/DefaultPreprocessor';
import type { FormField } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/MainModelDefaultSettings';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useEffect } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';
import type { ControlLoRAModelConfig, ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';
import { isAnimaControlNetModelConfig } from 'services/api/types';

export type ControlAdapterModelDefaultSettingsFormData = {
  preprocessor: FormField<string>;
  fp8Storage: FormField<boolean>;
};

type Props = {
  modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | ControlLoRAModelConfig;
};

// Only offer FP8 storage where a loader actually applies it, so the toggle never renders as a
// no-op. ControlLoRAs are patched into the base model rather than run standalone. Anima's LLLite
// adapters go through `AnimaControlNetLLLiteModel`, which never calls the layerwise cast - at
// 16-63MB per adapter there is nothing worth wiring up.
const supportsFp8Storage = (modelConfig: Props['modelConfig']): boolean =>
  modelConfig.type !== 'control_lora' && !isAnimaControlNetModelConfig(modelConfig);

export const ControlAdapterModelDefaultSettings = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const canManageModels = useIsModelManagerEnabled();

  const defaultSettingsDefaults = useControlAdapterModelDefaultSettings(modelConfig);

  const [updateModel, { isLoading: isLoadingUpdateModel }] = useUpdateModelMutation();

  const { handleSubmit, control, formState, reset } = useForm<ControlAdapterModelDefaultSettingsFormData>({
    defaultValues: defaultSettingsDefaults,
  });

  useEffect(() => {
    reset(defaultSettingsDefaults);
  }, [defaultSettingsDefaults, reset]);

  const onSubmit = useCallback<SubmitHandler<ControlAdapterModelDefaultSettingsFormData>>(
    (data) => {
      const body = {
        preprocessor: data.preprocessor.isEnabled ? data.preprocessor.value : null,
        // Null it out wherever the control is hidden. react-hook-form keeps unrendered fields in
        // `defaultValues`, so without this a value persisted before the control was hidden would be
        // re-sent verbatim on every save, with no UI left to clear it.
        fp8_storage: supportsFp8Storage(modelConfig) && data.fp8Storage.isEnabled ? data.fp8Storage.value : null,
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
    [updateModel, modelConfig, t, reset]
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

      <SimpleGrid columns={2} gap={8}>
        <DefaultPreprocessor control={control} name="preprocessor" />
        {supportsFp8Storage(modelConfig) && <DefaultFp8StorageControlAdapter control={control} name="fp8Storage" />}
      </SimpleGrid>
    </>
  );
});

ControlAdapterModelDefaultSettings.displayName = 'ControlAdapterModelDefaultSettings';
