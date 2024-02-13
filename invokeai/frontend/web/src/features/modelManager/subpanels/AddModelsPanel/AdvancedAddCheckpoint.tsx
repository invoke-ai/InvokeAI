import { Button, Checkbox, Flex, FormControl, FormErrorMessage, FormLabel, Input } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { setAdvancedAddScanModel } from 'features/modelManager/store/modelManagerSlice';
import BaseModelSelect from 'features/modelManager/subpanels/shared/BaseModelSelect';
import CheckpointConfigsSelect from 'features/modelManager/subpanels/shared/CheckpointConfigsSelect';
import ModelVariantSelect from 'features/modelManager/subpanels/shared/ModelVariantSelect';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import type { CSSProperties, FocusEventHandler } from 'react';
import { memo, useCallback, useState } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { useAddMainModelsMutation } from 'services/api/endpoints/models';
import type { CheckpointModelConfig } from 'services/api/types';

import { getModelName } from './util';

type AdvancedAddCheckpointProps = {
  model_path?: string;
};

const AdvancedAddCheckpoint = (props: AdvancedAddCheckpointProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { model_path } = props;

  const {
    register,
    handleSubmit,
    control,
    getValues,
    setValue,
    formState: { errors },
    reset,
  } = useForm<CheckpointModelConfig>({
    defaultValues: {
      model_name: model_path ? getModelName(model_path) : '',
      base_model: 'sd-1',
      model_type: 'main',
      path: model_path ? model_path : '',
      description: '',
      model_format: 'checkpoint',
      error: undefined,
      vae: '',
      variant: 'normal',
      config: 'configs\\stable-diffusion\\v1-inference.yaml',
    },
    mode: 'onChange',
  });

  const [addMainModel] = useAddMainModelsMutation();

  const [useCustomConfig, setUseCustomConfig] = useState<boolean>(false);

  const onSubmit = useCallback<SubmitHandler<CheckpointModelConfig>>(
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
                  modelName: values.model_name,
                }),
                status: 'success',
              })
            )
          );
          reset();

          // Close Advanced Panel in Scan Models tab
          if (model_path) {
            dispatch(setAdvancedAddScanModel(null));
          }
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
    [addMainModel, dispatch, model_path, reset, t]
  );

  const onBlur: FocusEventHandler<HTMLInputElement> = useCallback(
    (e) => {
      if (getValues().model_name === '') {
        const modelName = getModelName(e.currentTarget.value);
        if (modelName) {
          setValue('model_name', modelName as string);
        }
      }
    },
    [getValues, setValue]
  );

  const handleChangeUseCustomConfig = useCallback(() => setUseCustomConfig((prev) => !prev), []);

  return (
    <form onSubmit={handleSubmit(onSubmit)} style={formStyles}>
      <Flex flexDirection="column" gap={2}>
        <FormControl isInvalid={Boolean(errors.model_name)}>
          <FormLabel>{t('modelManager.model')}</FormLabel>
          <Input
            {...register('model_name', {
              validate: (value) => value.trim().length > 3 || 'Must be at least 3 characters',
            })}
          />
          {errors.model_name?.message && <FormErrorMessage>{errors.model_name?.message}</FormErrorMessage>}
        </FormControl>
        <BaseModelSelect<CheckpointModelConfig> control={control} name="base_model" />
        <FormControl isInvalid={Boolean(errors.path)}>
          <FormLabel>{t('modelManager.modelLocation')}</FormLabel>
          <Input
            {...register('path', {
              validate: (value) => value.trim().length > 0 || 'Must provide a path',
              onBlur,
            })}
          />
          {errors.path?.message && <FormErrorMessage>{errors.path?.message}</FormErrorMessage>}
        </FormControl>
        <FormControl>
          <FormLabel>{t('modelManager.description')}</FormLabel>
          <Input {...register('description')} />
        </FormControl>
        <FormControl>
          <FormLabel>{t('modelManager.vaeLocation')}</FormLabel>
          <Input {...register('vae')} />
        </FormControl>
        <ModelVariantSelect<CheckpointModelConfig> control={control} name="variant" />
        <Flex flexDirection="column" width="100%" gap={2}>
          {!useCustomConfig ? (
            <CheckpointConfigsSelect control={control} name="config" />
          ) : (
            <FormControl isRequired>
              <FormLabel>{t('modelManager.config')}</FormLabel>
              <Input {...register('config')} />
            </FormControl>
          )}
          <FormControl>
            <FormLabel>{t('modelManager.useCustomConfig')}</FormLabel>
            <Checkbox isChecked={useCustomConfig} onChange={handleChangeUseCustomConfig} />
          </FormControl>
          <Button mt={2} type="submit">
            {t('modelManager.addModel')}
          </Button>
        </Flex>
      </Flex>
    </form>
  );
};

const formStyles: CSSProperties = {
  width: '100%',
};

export default memo(AdvancedAddCheckpoint);
