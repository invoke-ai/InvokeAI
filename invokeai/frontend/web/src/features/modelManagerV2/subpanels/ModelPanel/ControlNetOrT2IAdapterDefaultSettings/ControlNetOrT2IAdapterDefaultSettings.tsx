import { Button, Flex, Heading, SimpleGrid, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useControlNetOrT2IAdapterDefaultSettings } from 'features/modelManagerV2/hooks/useControlNetOrT2IAdapterDefaultSettings';
import { DefaultPreprocessor } from 'features/modelManagerV2/subpanels/ModelPanel/ControlNetOrT2IAdapterDefaultSettings/DefaultPreprocessor';
import type { FormField } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/MainModelDefaultSettings';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';

export type ControlNetOrT2IAdapterDefaultSettingsFormData = {
  preprocessor: FormField<string>;
};

export const ControlNetOrT2IAdapterDefaultSettings = () => {
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { t } = useTranslation();

  const { defaultSettingsDefaults, isLoading: isLoadingDefaultSettings } =
    useControlNetOrT2IAdapterDefaultSettings(selectedModelKey);

  const [updateModel, { isLoading: isLoadingUpdateModel }] = useUpdateModelMutation();

  const { handleSubmit, control, formState, reset } = useForm<ControlNetOrT2IAdapterDefaultSettingsFormData>({
    defaultValues: defaultSettingsDefaults,
  });

  const onSubmit = useCallback<SubmitHandler<ControlNetOrT2IAdapterDefaultSettingsFormData>>(
    (data) => {
      if (!selectedModelKey) {
        return;
      }

      const body = {
        preprocessor: data.preprocessor.isEnabled ? data.preprocessor.value : null,
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
        <DefaultPreprocessor control={control} name="preprocessor" />
      </SimpleGrid>
    </>
  );
};
