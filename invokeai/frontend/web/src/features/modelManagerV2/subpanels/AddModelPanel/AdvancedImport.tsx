import { Button, Flex, FormControl, FormErrorMessage, FormLabel, Input, Text, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import BaseModelSelect from 'features/modelManagerV2/subpanels/ModelPanel/Fields/BaseModelSelect';
import BooleanSelect from 'features/modelManagerV2/subpanels/ModelPanel/Fields/BooleanSelect';
import ModelFormatSelect from 'features/modelManagerV2/subpanels/ModelPanel/Fields/ModelFormatSelect';
import ModelTypeSelect from 'features/modelManagerV2/subpanels/ModelPanel/Fields/ModelTypeSelect';
import ModelVariantSelect from 'features/modelManagerV2/subpanels/ModelPanel/Fields/ModelVariantSelect';
import PredictionTypeSelect from 'features/modelManagerV2/subpanels/ModelPanel/Fields/PredictionTypeSelect';
import RepoVariantSelect from 'features/modelManagerV2/subpanels/ModelPanel/Fields/RepoVariantSelect';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback, useEffect } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { useImportAdvancedModelMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

export const AdvancedImport = () => {
  const dispatch = useAppDispatch();

  const [importAdvancedModel] = useImportAdvancedModelMutation();

  const { t } = useTranslation();

  const {
    register,
    handleSubmit,
    control,
    formState: { errors },
    setValue,
    resetField,
    reset,
    watch,
  } = useForm<AnyModelConfig>({
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

  const onSubmit = useCallback<SubmitHandler<AnyModelConfig>>(
    (values) => {
      const cleanValues = Object.fromEntries(
        Object.entries(values).filter(([value]) => value !== null && value !== undefined)
      );
      importAdvancedModel({
        source: {
          path: cleanValues.path as string,
          type: 'local',
        },
        config: cleanValues,
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
    [dispatch, reset, t, importAdvancedModel]
  );

  const watchedModelType = watch('type');
  const watchedModelFormat = watch('format');

  useEffect(() => {
    if (watchedModelType === 'main') {
      setValue('format', 'diffusers');
      setValue('repo_variant', '');
      setValue('variant', 'normal');
    }
    if (watchedModelType === 'lora') {
      setValue('format', 'lycoris');
    } else if (watchedModelType === 'embedding') {
      setValue('format', 'embedding_file');
    } else if (watchedModelType === 'ip_adapter') {
      setValue('format', 'invokeai');
    } else {
      setValue('format', 'diffusers');
    }
    resetField('upcast_attention');
    resetField('ztsnr_training');
    resetField('vae');
    resetField('config');
    resetField('prediction_type');
    resetField('image_encoder_model_id');
  }, [watchedModelType, resetField, setValue]);

  return (
    <ScrollableContent>
      <form onSubmit={handleSubmit(onSubmit)}>
        <Flex flexDirection="column" gap={4} width="100%" pb={10}>
          <Flex alignItems="flex-end" gap="4">
            <FormControl flexDir="column" alignItems="flex-start" gap={1}>
              <FormLabel>{t('modelManager.modelType')}</FormLabel>
              <ModelTypeSelect<AnyModelConfig> control={control} name="type" />
            </FormControl>
            <Text px="2" fontSize="xs" textAlign="center">
              {t('modelManager.advancedImportInfo')}
            </Text>
          </Flex>

          <Flex p={4} borderRadius={4} bg="base.850" height="100%" direction="column" gap="3">
            <FormControl isInvalid={Boolean(errors.name)}>
              <Flex direction="column" width="full">
                <FormLabel>{t('modelManager.name')}</FormLabel>
                <Input
                  {...register('name', {
                    validate: (value) => value.trim().length >= 3 || 'Must be at least 3 characters',
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
            <Flex gap={4}>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.baseModel')}</FormLabel>
                <BaseModelSelect<AnyModelConfig> control={control} name="base" />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('common.format')}</FormLabel>
                <ModelFormatSelect<AnyModelConfig> control={control} name="format" />
              </FormControl>
            </Flex>
            <Flex gap={4}>
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
                    <FormLabel>{t('modelManager.vaeLocation')}</FormLabel>
                    <Input {...register('vae')} />
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
            <Button mt={2} type="submit">
              {t('modelManager.addModel')}
            </Button>
          </Flex>
        </Flex>
      </form>
    </ScrollableContent>
  );
};
