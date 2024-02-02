import { Button, Flex, FormControl, FormErrorMessage, FormLabel, Input } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { setAdvancedAddScanModel } from 'features/modelManager/store/modelManagerSlice';
import BaseModelSelect from 'features/modelManager/subpanels/shared/BaseModelSelect';
import ModelVariantSelect from 'features/modelManager/subpanels/shared/ModelVariantSelect';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import type { CSSProperties, FocusEventHandler } from 'react';
import { memo, useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { useAddMainModelsMutation } from 'services/api/endpoints/models';
import type { DiffusersModelConfig } from 'services/api/types';

import { getModelName } from './util';

type AdvancedAddDiffusersProps = {
  model_path?: string;
};

const AdvancedAddDiffusers = (props: AdvancedAddDiffusersProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { model_path } = props;

  const [addMainModel] = useAddMainModelsMutation();

  const {
    register,
    handleSubmit,
    control,
    getValues,
    setValue,
    formState: { errors },
    reset,
  } = useForm<DiffusersModelConfig>({
    defaultValues: {
      model_name: model_path ? getModelName(model_path, false) : '',
      base_model: 'sd-1',
      model_type: 'main',
      path: model_path ? model_path : '',
      description: '',
      model_format: 'diffusers',
      error: undefined,
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
        const modelName = getModelName(e.currentTarget.value, false);
        if (modelName) {
          setValue('model_name', modelName as string);
        }
      }
    },
    [getValues, setValue]
  );

  return (
    <form onSubmit={handleSubmit(onSubmit)} style={formStyles}>
      <Flex flexDirection="column" gap={2}>
        <FormControl isInvalid={Boolean(errors.model_name)}>
          <FormLabel>{t('modelManager.name')}</FormLabel>
          <Input
            {...register('model_name', {
              validate: (value) => value.trim().length > 3 || 'Must be at least 3 characters',
            })}
          />
          {errors.model_name?.message && <FormErrorMessage>{errors.model_name?.message}</FormErrorMessage>}
        </FormControl>
        <BaseModelSelect<DiffusersModelConfig> control={control} name="base_model" />
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
        <ModelVariantSelect<DiffusersModelConfig> control={control} name="variant" />

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

export default memo(AdvancedAddDiffusers);
