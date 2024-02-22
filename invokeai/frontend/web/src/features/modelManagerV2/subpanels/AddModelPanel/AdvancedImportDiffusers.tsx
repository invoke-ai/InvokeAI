import { Button, Flex, FormControl, FormErrorMessage, FormLabel, Input, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import type { CSSProperties } from 'react';
import {useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { useAddMainModelsMutation } from 'services/api/endpoints/models';
import type { DiffusersModelConfig } from 'services/api/types';

import BaseModelSelect from './BaseModelSelect';
import ModelVariantSelect from './ModelVariantSelect';

export const AdvancedImportDiffusers = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const [addMainModel] = useAddMainModelsMutation();

  const {
    register,
    handleSubmit,
    control,
    formState: { errors },
    reset,
  } = useForm<DiffusersModelConfig>({
    defaultValues: {
      name: '',
      base: 'sd-1',
      type: 'main',
      path: '',
      description: '',
      format: 'diffusers',
      vae: '',
      variant: 'normal',
    },
    mode: 'onChange',
  });

  const onSubmit = useCallback<SubmitHandler<DiffusersModelConfig>>(
    (values) => {
      addMainModel({
        body: values,
      })
        .unwrap()
        .then((_) => {
          dispatch(
            addToast(
              makeToast({
                title: t('modelManager.modelAdded', {
                  modelName: values.name,
                }),
                status: 'success',
              })
            )
          );
          reset();
        })
        .catch((error) => {
          if (error) {
            dispatch(
              addToast(
                makeToast({
                  title: t('toast.modelAddFailed'),
                  status: 'error',
                })
              )
            );
          }
        });
    },
    [addMainModel, dispatch, reset, t]
  );

  return (
      <form onSubmit={handleSubmit(onSubmit)} style={formStyles}>
        <Flex flexDirection="column" gap={2}>
          <FormControl isInvalid={Boolean(errors.name)}>
            <Flex direction="column" width="full">
              <FormLabel>{t('modelManager.name')}</FormLabel>
              <Input
                {...register('name', {
                  validate: (value) => value.trim().length > 3 || 'Must be at least 3 characters',
                })}
              />
              {errors.name?.message && <FormErrorMessage>{errors.name?.message}</FormErrorMessage>}
            </Flex>
          </FormControl>
          <Flex>
            <FormControl>
              <Flex direction="column" width="full">
                <FormLabel>{t('modelManager.description')}</FormLabel>
                <Textarea size="sm" {...register('description')} />
                {errors.name?.message && <FormErrorMessage>{errors.name?.message}</FormErrorMessage>}
              </Flex>
            </FormControl>
          </Flex>
          <Flex gap="2">
            <BaseModelSelect<DiffusersModelConfig> control={control} name="base" />
            <ModelVariantSelect<DiffusersModelConfig> control={control} name="variant" />
          </Flex>
          <FormControl isInvalid={Boolean(errors.path)}>
            <Flex direction="column" width="full">
              <FormLabel>{t('modelManager.modelLocation')}</FormLabel>
              <Input
                {...register('path', {
                  validate: (value) => value.trim().length > 0 || 'Must provide a path',
                })}
              />
              {errors.path?.message && <FormErrorMessage>{errors.path?.message}</FormErrorMessage>}
            </Flex>
          </FormControl>
          <FormControl>
            <Flex direction="column" width="full">
              <FormLabel>{t('modelManager.vaeLocation')}</FormLabel>
              <Input {...register('vae')} />
            </Flex>
          </FormControl>

          <Button mt={2} type="submit">
            {t('modelManager.addModel')}
          </Button>
        </Flex>
      </form>
  );
};

const formStyles: CSSProperties = {
  width: '100%',
};
