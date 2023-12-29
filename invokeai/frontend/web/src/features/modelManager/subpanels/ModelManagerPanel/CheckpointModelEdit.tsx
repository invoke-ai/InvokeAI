import { Badge, Divider, Flex } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvCheckbox } from 'common/components/InvCheckbox/wrapper';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvInput } from 'common/components/InvInput/InvInput';
import { InvText } from 'common/components/InvText/wrapper';
import BaseModelSelect from 'features/modelManager/subpanels/shared/BaseModelSelect';
import CheckpointConfigsSelect from 'features/modelManager/subpanels/shared/CheckpointConfigsSelect';
import ModelVariantSelect from 'features/modelManager/subpanels/shared/ModelVariantSelect';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { memo, useCallback, useEffect, useState } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import type { CheckpointModelConfigEntity } from 'services/api/endpoints/models';
import {
  useGetCheckpointConfigsQuery,
  useUpdateMainModelsMutation,
} from 'services/api/endpoints/models';
import type { CheckpointModelConfig } from 'services/api/types';

import ModelConvert from './ModelConvert';

type CheckpointModelEditProps = {
  model: CheckpointModelConfigEntity;
};

const CheckpointModelEdit = (props: CheckpointModelEditProps) => {
  const { model } = props;

  const [updateMainModel, { isLoading }] = useUpdateMainModelsMutation();
  const { data: availableCheckpointConfigs } = useGetCheckpointConfigsQuery();

  const [useCustomConfig, setUseCustomConfig] = useState<boolean>(false);

  useEffect(() => {
    if (!availableCheckpointConfigs?.includes(model.config)) {
      setUseCustomConfig(true);
    }
  }, [availableCheckpointConfigs, model.config]);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const {
    register,
    handleSubmit,
    control,
    formState: { errors },
    reset,
  } = useForm<CheckpointModelConfig>({
    defaultValues: {
      model_name: model.model_name ? model.model_name : '',
      base_model: model.base_model,
      model_type: 'main',
      path: model.path ? model.path : '',
      description: model.description ? model.description : '',
      model_format: 'checkpoint',
      vae: model.vae ? model.vae : '',
      config: model.config ? model.config : '',
      variant: model.variant,
    },
    mode: 'onChange',
  });

  const handleChangeUseCustomConfig = useCallback(
    () => setUseCustomConfig((prev) => !prev),
    []
  );

  const onSubmit = useCallback<SubmitHandler<CheckpointModelConfig>>(
    (values) => {
      const responseBody = {
        base_model: model.base_model,
        model_name: model.model_name,
        body: values,
      };
      updateMainModel(responseBody)
        .unwrap()
        .then((payload) => {
          reset(payload as CheckpointModelConfig, { keepDefaultValues: true });
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
    [dispatch, model.base_model, model.model_name, reset, t, updateMainModel]
  );

  return (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex justifyContent="space-between" alignItems="center">
        <Flex flexDirection="column">
          <InvText fontSize="lg" fontWeight="bold">
            {model.model_name}
          </InvText>
          <InvText fontSize="sm" color="base.400">
            {MODEL_TYPE_MAP[model.base_model]} {t('modelManager.model')}
          </InvText>
        </Flex>
        {![''].includes(model.base_model) ? (
          <ModelConvert model={model} />
        ) : (
          <Badge p={2} borderRadius={4} bg="error.400">
            {t('modelManager.conversionNotSupported')}
          </Badge>
        )}
      </Flex>
      <Divider />

      <Flex
        flexDirection="column"
        maxHeight={window.innerHeight - 270}
        overflowY="scroll"
      >
        <form onSubmit={handleSubmit(onSubmit)}>
          <Flex flexDirection="column" overflowY="scroll" gap={4}>
            <InvControl
              label={t('modelManager.name')}
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
            <InvControl label={t('modelManager.description')}>
              <InvInput {...register('description')} />
            </InvControl>
            <BaseModelSelect<CheckpointModelConfig>
              control={control}
              name="base_model"
            />
            <ModelVariantSelect<CheckpointModelConfig>
              control={control}
              name="variant"
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
                })}
              />
            </InvControl>
            <InvControl label={t('modelManager.vaeLocation')}>
              <InvInput {...register('vae')} />
            </InvControl>

            <Flex flexDirection="column" gap={2}>
              {!useCustomConfig ? (
                <CheckpointConfigsSelect control={control} name="config" />
              ) : (
                <InvControl isRequired label={t('modelManager.config')}>
                  <InvInput {...register('config')} />
                </InvControl>
              )}
              <InvControl label="Use Custom Config">
                <InvCheckbox
                  isChecked={useCustomConfig}
                  onChange={handleChangeUseCustomConfig}
                />
              </InvControl>
            </Flex>

            <InvButton type="submit" isLoading={isLoading}>
              {t('modelManager.updateModel')}
            </InvButton>
          </Flex>
        </form>
      </Flex>
    </Flex>
  );
};

export default memo(CheckpointModelEdit);
