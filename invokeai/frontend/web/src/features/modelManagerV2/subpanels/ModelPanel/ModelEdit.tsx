import {
  Button,
  Flex,
  FormControl,
  FormErrorMessage,
  FormLabel,
  Heading,
  Input,
  Text,
  Textarea,
} from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setSelectedModelMode } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import type { UpdateModelArg } from 'services/api/endpoints/models';
import { useGetModelConfigQuery, useUpdateModelsMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

import BaseModelSelect from './Fields/BaseModelSelect';
import BooleanSelect from './Fields/BooleanSelect';
import ModelFormatSelect from './Fields/ModelFormatSelect';
import ModelTypeSelect from './Fields/ModelTypeSelect';
import ModelVariantSelect from './Fields/ModelVariantSelect';
import PredictionTypeSelect from './Fields/PredictionTypeSelect';
import RepoVariantSelect from './Fields/RepoVariantSelect';
import VaeSelect from './Fields/VaeSelect';

export const ModelEdit = () => {
  const dispatch = useAppDispatch();
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data, isLoading } = useGetModelConfigQuery(selectedModelKey ?? skipToken);

  const [updateModel, { isLoading: isSubmitting }] = useUpdateModelsMutation();

  const { t } = useTranslation();

  const {
    register,
    handleSubmit,
    control,
    formState: { errors },
    reset,
    watch,
  } = useForm<UpdateModelArg['body']>({
    defaultValues: {
      ...data,
    },
    mode: 'onChange',
  });

  const watchedModelType = watch('type');
  const watchedModelFormat = watch('format');

  const onSubmit = useCallback<SubmitHandler<AnyModelConfig>>(
    (values) => {
      console.log({ values });
      if (!data?.key) {
        return;
      }

      const responseBody: UpdateModelArg = {
        key: data.key,
        body: values,
      };

      updateModel(responseBody)
        .unwrap()
        .then((payload) => {
          reset(payload, { keepDefaultValues: true });
          dispatch(setSelectedModelMode('view'));
          dispatch(
            addToast(
              makeToast({
                title: t('modelManager.modelUpdated'),
                status: 'success',
              })
            )
          );
        })
        .catch((_) => {
          reset();
          dispatch(
            addToast(
              makeToast({
                title: t('modelManager.modelUpdateFailed'),
                status: 'error',
              })
            )
          );
        });
    },
    [dispatch, data?.key, reset, t, updateModel]
  );

  const handleClickCancel = useCallback(() => {
    dispatch(setSelectedModelMode('view'));
  }, [dispatch]);

  if (isLoading) {
    return <Text>{t('common.loading')}</Text>;
  }

  if (!data) {
    return <Text>{t('common.somethingWentWrong')}</Text>;
  }
  return (
    <Flex flexDir="column" h="full">
      <form onSubmit={handleSubmit(onSubmit)}>
        <FormControl flexDir="column" alignItems="flex-start" gap={1} isInvalid={Boolean(errors.name)}>
          <Flex w="full" justifyContent="space-between" gap={4} alignItems="center">
            <FormLabel hidden={true}>{t('modelManager.modelName')}</FormLabel>
            <Input
              {...register('name', {
                validate: (value) => value.trim().length > 3 || 'Must be at least 3 characters',
              })}
              size="lg"
            />

            <Flex gap={2}>
              <Button size="sm" onClick={handleClickCancel}>
                {t('common.cancel')}
              </Button>
              <Button
                size="sm"
                colorScheme="invokeYellow"
                onClick={handleSubmit(onSubmit)}
                isLoading={isSubmitting}
                isDisabled={Boolean(Object.keys(errors).length)}
              >
                {t('common.save')}
              </Button>
            </Flex>
          </Flex>
          {errors.name?.message && <FormErrorMessage>{errors.name?.message}</FormErrorMessage>}
        </FormControl>

        <Flex flexDir="column" gap={3} mt="4">
          <Flex>
            <FormControl flexDir="column" alignItems="flex-start" gap={1}>
              <FormLabel>{t('modelManager.description')}</FormLabel>
              <Textarea fontSize="md" resize="none" {...register('description')} />
            </FormControl>
          </Flex>
          <Heading as="h3" fontSize="md" mt="4">
            {t('modelManager.modelSettings')}
          </Heading>
          <Flex gap={4}>
            <FormControl flexDir="column" alignItems="flex-start" gap={1}>
              <FormLabel>{t('modelManager.baseModel')}</FormLabel>
              <BaseModelSelect control={control} name="base" />
            </FormControl>
            <FormControl flexDir="column" alignItems="flex-start" gap={1}>
              <FormLabel>{t('modelManager.modelType')}</FormLabel>
              <ModelTypeSelect<AnyModelConfig> control={control} name="type" />
            </FormControl>
          </Flex>
          <Flex gap={4}>
            <FormControl flexDir="column" alignItems="flex-start" gap={1}>
              <FormLabel>{t('common.format')}</FormLabel>
              <ModelFormatSelect control={control} name="format" />
            </FormControl>
            <FormControl flexDir="column" alignItems="flex-start" gap={1} isInvalid={Boolean(errors.path)}>
              <FormLabel>{t('modelManager.path')}</FormLabel>
              <Input
                {...register('path', {
                  validate: (value) => value.trim().length > 0 || 'Must provide a path',
                })}
              />
              {errors.path?.message && <FormErrorMessage>{errors.path?.message}</FormErrorMessage>}
            </FormControl>
          </Flex>
          {watchedModelType === 'main' && (
            <>
              <Flex gap={4}>
                {watchedModelFormat === 'diffusers' && (
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.repoVariant')}</FormLabel>
                    <RepoVariantSelect<AnyModelConfig> control={control} name="repo_variant" />
                  </FormControl>
                )}
                {watchedModelFormat === 'checkpoint' && (
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.pathToConfig')}</FormLabel>
                    <Input {...register('config')} />
                  </FormControl>
                )}

                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.variant')}</FormLabel>
                  <ModelVariantSelect<AnyModelConfig> control={control} name="variant" />
                </FormControl>
              </Flex>
              <Flex gap={4}>
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.predictionType')}</FormLabel>
                  <PredictionTypeSelect<AnyModelConfig> control={control} name="prediction_type" />
                </FormControl>
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.upcastAttention')}</FormLabel>
                  <BooleanSelect<AnyModelConfig> control={control} name="upcast_attention" />
                </FormControl>
              </Flex>
              <Flex gap={4}>
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.ztsnrTraining')}</FormLabel>
                  <BooleanSelect<AnyModelConfig> control={control} name="ztsnr_training" />
                </FormControl>
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.vae')}</FormLabel>
                  <VaeSelect control={control} name="vae" />
                </FormControl>
              </Flex>
            </>
          )}
          {watchedModelType === 'ip_adapter' && (
            <Flex gap={4}>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.imageEncoderModelId')}</FormLabel>
                <Input {...register('image_encoder_model_id')} />
              </FormControl>
            </Flex>
          )}
        </Flex>
      </form>
    </Flex>
  );
};
