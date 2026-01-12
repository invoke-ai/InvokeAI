import { Button, Flex, Heading, SimpleGrid } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { useLoRAModelDefaultSettings } from 'features/modelManagerV2/hooks/useLoRAModelDefaultSettings';
import { DefaultWeight } from 'features/modelManagerV2/subpanels/ModelPanel/LoRAModelDefaultSettings/DefaultWeight';
import type { FormField } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/MainModelDefaultSettings';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';
import type { LoRAModelConfig } from 'services/api/types';

export type LoRAModelDefaultSettingsFormData = {
  weight: FormField<number>;
};

type Props = {
  modelConfig: LoRAModelConfig;
};

export const LoRAModelDefaultSettings = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const user = useAppSelector(selectCurrentUser);

  // Only admins can save model default settings
  const isAdmin = user?.is_admin ?? false;

  const defaultSettingsDefaults = useLoRAModelDefaultSettings(modelConfig);

  const [updateModel, { isLoading: isLoadingUpdateModel }] = useUpdateModelMutation();

  const { handleSubmit, control, formState, reset } = useForm<LoRAModelDefaultSettingsFormData>({
    defaultValues: defaultSettingsDefaults,
  });

  const onSubmit = useCallback<SubmitHandler<LoRAModelDefaultSettingsFormData>>(
    (data) => {
      const body = {
        weight: data.weight.isEnabled ? data.weight.value : null,
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
        {isAdmin && (
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
        <DefaultWeight control={control} name="weight" />
      </SimpleGrid>
    </>
  );
});

LoRAModelDefaultSettings.displayName = 'LoRAModelDefaultSettings';
