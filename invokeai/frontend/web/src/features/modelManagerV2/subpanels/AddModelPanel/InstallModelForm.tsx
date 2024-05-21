import { Button, Checkbox, Flex, FormControl, FormHelperText, FormLabel, Input } from '@invoke-ai/ui-library';
import { toast, ToastID } from 'features/toast/toast';
import { t } from 'i18next';
import { useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useInstallModelMutation } from 'services/api/endpoints/models';

type SimpleImportModelConfig = {
  location: string;
  inplace: boolean;
};

export const InstallModelForm = () => {
  const [installModel, { isLoading }] = useInstallModelMutation();

  const { register, handleSubmit, formState, reset } = useForm<SimpleImportModelConfig>({
    defaultValues: {
      location: '',
      inplace: true,
    },
    mode: 'onChange',
  });

  const onSubmit = useCallback<SubmitHandler<SimpleImportModelConfig>>(
    (values) => {
      if (!values?.location) {
        return;
      }

      installModel({ source: values.location, inplace: values.inplace })
        .unwrap()
        .then((_) => {
          toast({
            id: ToastID.MODEL_INSTALL_QUEUED,
            title: t('toast.modelAddedSimple'),
            status: 'success',
          });
          reset(undefined, { keepValues: true });
        })
        .catch((error) => {
          reset(undefined, { keepValues: true });
          if (error) {
            toast({
              id: ToastID.MODEL_INSTALL_QUEUE_FAILED,
              title: `${error.data.detail} `,
              status: 'error',
            });
          }
        });
    },
    [reset, installModel]
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
};
