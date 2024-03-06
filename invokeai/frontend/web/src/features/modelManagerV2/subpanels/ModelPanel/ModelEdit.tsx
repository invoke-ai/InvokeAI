import {
  Button,
  Checkbox,
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
import { useGetModelConfigQuery, useUpdateModelMutation } from 'services/api/endpoints/models';

import BaseModelSelect from './Fields/BaseModelSelect';
import ModelVariantSelect from './Fields/ModelVariantSelect';
import PredictionTypeSelect from './Fields/PredictionTypeSelect';

export const ModelEdit = () => {
  const dispatch = useAppDispatch();
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data, isLoading } = useGetModelConfigQuery(selectedModelKey ?? skipToken);

  const [updateModel, { isLoading: isSubmitting }] = useUpdateModelMutation();

  const { t } = useTranslation();

  const {
    register,
    handleSubmit,
    control,
    formState: { errors },
    reset,
  } = useForm<UpdateModelArg['body']>({
    defaultValues: {
      ...data,
    },
    mode: 'onChange',
  });

  const onSubmit = useCallback<SubmitHandler<UpdateModelArg['body']>>(
    (values) => {
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
        <Flex w="full" justifyContent="space-between" gap={4} alignItems="center">
          <FormControl flexDir="column" alignItems="flex-start" gap={1} isInvalid={Boolean(errors.name)}>
            <FormLabel hidden={true}>{t('modelManager.modelName')}</FormLabel>
            <Input
              {...register('name', {
                validate: (value) => (value && value.trim().length > 3) || 'Must be at least 3 characters',
              })}
              size="lg"
            />

            {errors.name?.message && <FormErrorMessage>{errors.name?.message}</FormErrorMessage>}
          </FormControl>
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
              <BaseModelSelect control={control} />
            </FormControl>
          </Flex>
          {data.type === 'main' && data.format === 'checkpoint' && (
            <>
              <Flex gap={4}>
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.pathToConfig')}</FormLabel>
                  <Input
                    {...register('config_path', {
                      validate: (value) => (value && value.trim().length > 3) || 'Must be at least 3 characters',
                    })}
                  />
                </FormControl>
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.variant')}</FormLabel>
                  <ModelVariantSelect control={control} />
                </FormControl>
              </Flex>
              <Flex gap={4}>
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.predictionType')}</FormLabel>
                  <PredictionTypeSelect control={control} />
                </FormControl>
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.upcastAttention')}</FormLabel>
                  <Checkbox {...register('upcast_attention')} />
                </FormControl>
              </Flex>
            </>
          )}
        </Flex>
      </form>
    </Flex>
  );
};
