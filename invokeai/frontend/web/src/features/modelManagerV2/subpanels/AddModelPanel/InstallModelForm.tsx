import { Button, Checkbox, Flex, FormControl, FormHelperText, FormLabel, Input } from '@invoke-ai/ui-library';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import { t } from 'i18next';
import { memo, useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';

type SimpleImportModelConfig = {
  location: string;
  inplace: boolean;
};

export const InstallModelForm = memo(() => {
  const [installModel, { isLoading }] = useInstallModel();

  const { register, handleSubmit, formState, reset } = useForm<SimpleImportModelConfig>({
    defaultValues: {
      location: '',
      inplace: true,
    },
    mode: 'onChange',
  });

  const resetForm = useCallback(() => reset(undefined, { keepValues: true }), [reset]);

  const onSubmit = useCallback<SubmitHandler<SimpleImportModelConfig>>(
    (values) => {
      if (!values?.location) {
        return;
      }

      installModel({
        source: values.location,
        inplace: values.inplace,
        onSuccess: resetForm,
        onError: resetForm,
      });
    },
    [installModel, resetForm]
  );

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <Flex flexDir="column" gap={4}>
        <Flex gap={2} alignItems="flex-end" justifyContent="space-between">
          <FormControl orientation="vertical">
            <FormLabel>{t('modelManager.urlOrLocalPath')}</FormLabel>
            <Flex alignItems="center" gap={3} w="full">
              <Input placeholder={t('modelManager.simpleModelPlaceholder')} {...register('location')} />
              <Button
                onClick={handleSubmit(onSubmit)}
                isDisabled={!formState.dirtyFields.location}
                isLoading={isLoading}
                size="sm"
              >
                {t('modelManager.install')}
              </Button>
            </Flex>
            <FormHelperText>{t('modelManager.urlOrLocalPathHelper')}</FormHelperText>
          </FormControl>
        </Flex>

        <FormControl>
          <Flex flexDir="column" gap={2}>
            <Flex gap={4}>
              <Checkbox {...register('inplace')} />
              <FormLabel>
                {t('modelManager.inplaceInstall')} ({t('modelManager.localOnly')})
              </FormLabel>
            </Flex>
            <FormHelperText>{t('modelManager.inplaceInstallDesc')}</FormHelperText>
          </Flex>
        </FormControl>
      </Flex>
    </form>
  );
});

InstallModelForm.displayName = 'InstallModelForm';
