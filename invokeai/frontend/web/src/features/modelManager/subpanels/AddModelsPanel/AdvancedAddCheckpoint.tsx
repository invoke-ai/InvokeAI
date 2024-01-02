import { Flex } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvCheckbox } from 'common/components/InvCheckbox/wrapper';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvInput } from 'common/components/InvInput/InvInput';
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

  const handleChangeUseCustomConfig = useCallback(
    () => setUseCustomConfig((prev) => !prev),
    []
  );

  return (
    <form onSubmit={handleSubmit(onSubmit)} style={formStyles}>
      <Flex flexDirection="column" gap={2}>
        <InvControl
          label={t('modelManager.model')}
          isInvalid={Boolean(errors.model_name)}
          error={errors.model_name?.message}
        >
          <InvInput
            {...register('model_name', {
              validate: (value) =>
                value.trim().length > 3 || 'Must be at least 3 characters',
            })}
          />
        </InvControl>
        <BaseModelSelect<CheckpointModelConfig>
          control={control}
          name="base_model"
        />
        <InvControl
          label={t('modelManager.modelLocation')}
          isInvalid={Boolean(errors.path)}
          error={errors.path?.message}
        >
          <InvInput
            {...register('path', {
              validate: (value) =>
                value.trim().length > 0 || 'Must provide a path',
              onBlur,
            })}
          />
        </InvControl>
        <InvControl label={t('modelManager.description')}>
          <InvInput {...register('description')} />
        </InvControl>
        <InvControl label={t('modelManager.vaeLocation')}>
          <InvInput {...register('vae')} />
        </InvControl>
        <ModelVariantSelect<CheckpointModelConfig>
          control={control}
          name="variant"
        />
        <Flex flexDirection="column" width="100%" gap={2}>
          {!useCustomConfig ? (
            <CheckpointConfigsSelect control={control} name="config" />
          ) : (
            <InvControl isRequired label={t('modelManager.config')}>
              <InvInput {...register('config')} />
            </InvControl>
          )}
          <InvControl label={t('modelManager.useCustomConfig')}>
            <InvCheckbox
              isChecked={useCustomConfig}
              onChange={handleChangeUseCustomConfig}
            />
          </InvControl>
          <InvButton mt={2} type="submit">
            {t('modelManager.addModel')}
          </InvButton>
        </Flex>
      </Flex>
    </form>
  );
};

const formStyles: CSSProperties = {
  width: '100%',
};

export default memo(AdvancedAddCheckpoint);
