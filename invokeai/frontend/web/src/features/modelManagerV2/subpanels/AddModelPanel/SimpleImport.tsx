import { Button, Flex, FormControl, FormLabel, Input } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import { useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useImportMainModelsMutation } from 'services/api/endpoints/models';

type SimpleImportModelConfig = {
  location: string;
};

export const SimpleImport = () => {
  const dispatch = useAppDispatch();

  const [importMainModel, { isLoading }] = useImportMainModelsMutation();

  const { register, handleSubmit, formState, reset } = useForm<SimpleImportModelConfig>({
    defaultValues: {
      location: '',
    },
    mode: 'onChange',
  });

  const onSubmit = useCallback<SubmitHandler<SimpleImportModelConfig>>(
    (values) => {
      if (!values?.location) {
        return;
      }

      importMainModel({ source: values.location, config: undefined })
        .unwrap()
        .then((_) => {
          dispatch(
            addToast(
              makeToast({
                title: t('toast.modelAddedSimple'),
                status: 'success',
              })
            )
          );
          reset();
        })
        .catch((error) => {
          reset();
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
    [dispatch, reset, importMainModel]
  );

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <Flex gap={2} alignItems="flex-end" justifyContent="space-between">
        <FormControl>
          <Flex direction="column" w="full">
            <FormLabel>{t('modelManager.modelLocation')}</FormLabel>
            <Input {...register('location')} />
          </Flex>
        </FormControl>
        <Button onClick={handleSubmit(onSubmit)} isDisabled={!formState.isDirty} isLoading={isLoading} type="submit">
          {t('modelManager.addModel')}
        </Button>
      </Flex>
    </form>
  );
};
