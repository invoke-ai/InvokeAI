import { Button, Divider, Flex, FormControl, FormErrorMessage, FormLabel, Input, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import BaseModelSelect from 'features/modelManager/subpanels/shared/BaseModelSelect';
import { LORA_MODEL_FORMAT_MAP, MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { memo, useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import type { LoRAModelConfigEntity } from 'services/api/endpoints/models';
import { useUpdateLoRAModelsMutation } from 'services/api/endpoints/models';
import type { LoRAModelConfig } from 'services/api/types';

type LoRAModelEditProps = {
  model: LoRAModelConfigEntity;
};

const LoRAModelEdit = (props: LoRAModelEditProps) => {
  const { model } = props;

  const [updateLoRAModel, { isLoading }] = useUpdateLoRAModelsMutation();

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const {
    register,
    handleSubmit,
    control,
    formState: { errors },
    reset,
  } = useForm<LoRAModelConfig>({
    defaultValues: {
      model_name: model.model_name ? model.model_name : '',
      base_model: model.base_model,
      model_type: 'lora',
      path: model.path ? model.path : '',
      description: model.description ? model.description : '',
      model_format: model.model_format,
    },
    mode: 'onChange',
  });

  const onSubmit = useCallback<SubmitHandler<LoRAModelConfig>>(
    (values) => {
      const responseBody = {
        base_model: model.base_model,
        model_name: model.model_name,
        body: values,
      };

      updateLoRAModel(responseBody)
        .unwrap()
        .then((payload) => {
          reset(payload as LoRAModelConfig, { keepDefaultValues: true });
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
    [dispatch, model.base_model, model.model_name, reset, t, updateLoRAModel]
  );

  return (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex flexDirection="column">
        <Text fontSize="lg" fontWeight="bold">
          {model.model_name}
        </Text>
        <Text fontSize="sm" color="base.400">
          {MODEL_TYPE_MAP[model.base_model]} {t('modelManager.model')} ⋅ {LORA_MODEL_FORMAT_MAP[model.model_format]}{' '}
          {t('common.format')}
        </Text>
      </Flex>
      <Divider />

      <form onSubmit={handleSubmit(onSubmit)}>
        <Flex flexDirection="column" overflowY="scroll" gap={4}>
          <FormControl isInvalid={Boolean(errors.model_name)}>
            <FormLabel>{t('modelManager.name')}</FormLabel>
            <Input
              {...register('model_name', {
                validate: (value) => value.trim().length > 3 || 'Must be at least 3 characters',
              })}
            />
            {errors.model_name?.message && <FormErrorMessage>{errors.model_name?.message}</FormErrorMessage>}
          </FormControl>
          <FormControl>
            <FormLabel>{t('modelManager.description')}</FormLabel>
            <Input {...register('description')} />
          </FormControl>
          <BaseModelSelect<LoRAModelConfig> control={control} name="base_model" />

          <FormControl isInvalid={Boolean(errors.path)}>
            <FormLabel>{t('modelManager.modelLocation')}</FormLabel>
            <Input
              {...register('path', {
                validate: (value) => value.trim().length > 0 || 'Must provide a path',
              })}
            />
            {errors.path?.message && <FormErrorMessage>{errors.path?.message}</FormErrorMessage>}
          </FormControl>
          <Button type="submit" isLoading={isLoading}>
            {t('modelManager.updateModel')}
          </Button>
        </Flex>
      </form>
    </Flex>
  );
};

export default memo(LoRAModelEdit);
