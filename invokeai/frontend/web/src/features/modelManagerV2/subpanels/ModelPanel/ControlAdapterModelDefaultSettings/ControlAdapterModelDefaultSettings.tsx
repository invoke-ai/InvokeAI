import { Button, Flex, Heading, SimpleGrid } from '@invoke-ai/ui-library';
import { useControlAdapterModelDefaultSettings } from 'features/modelManagerV2/hooks/useControlAdapterModelDefaultSettings';
import { DefaultPreprocessor } from 'features/modelManagerV2/subpanels/ModelPanel/ControlAdapterModelDefaultSettings/DefaultPreprocessor';
import type { FormField } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/MainModelDefaultSettings';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';
import type { ControlLoRAModelConfig, ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';

export type ControlAdapterModelDefaultSettingsFormData = {
  preprocessor: FormField<string>;
};

type Props = {
  modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | ControlLoRAModelConfig;
};

export const ControlAdapterModelDefaultSettings = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();

  const defaultSettingsDefaults = useControlAdapterModelDefaultSettings(modelConfig);

  const [updateModel, { isLoading: isLoadingUpdateModel }] = useUpdateModelMutation();

  const { handleSubmit, control, formState, reset } = useForm<ControlAdapterModelDefaultSettingsFormData>({
    defaultValues: defaultSettingsDefaults,
  });

  const onSubmit = useCallback<SubmitHandler<ControlAdapterModelDefaultSettingsFormData>>(
    (data) => {
      const body = {
        preprocessor: data.preprocessor.isEnabled ? data.preprocessor.value : null,
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
    [updateModel, modelConfig.key, t, reset]
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
        <DefaultPreprocessor control={control} name="preprocessor" />
      </SimpleGrid>
    </>
  );
});

ControlAdapterModelDefaultSettings.displayName = 'ControlAdapterModelDefaultSettings';
