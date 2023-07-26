import { Badge, Divider, Flex, Text } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  CheckpointModelConfigEntity,
  useGetCheckpointConfigsQuery,
  useUpdateMainModelsMutation,
} from 'services/api/endpoints/models';
import { CheckpointModelConfig } from 'services/api/types';
import BaseModelSelect from '../shared/BaseModelSelect';
import CheckpointConfigsSelect from '../shared/CheckpointConfigsSelect';
import ModelVariantSelect from '../shared/ModelVariantSelect';
import ModelConvert from './ModelConvert';

type CheckpointModelEditProps = {
  model: CheckpointModelConfigEntity;
};

export default function CheckpointModelEdit(props: CheckpointModelEditProps) {
  const isBusy = useAppSelector(selectIsBusy);

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

  const checkpointEditForm = useForm<CheckpointModelConfig>({
    initialValues: {
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
    validate: {
      path: (value) =>
        value.trim().length === 0 ? 'Must provide a path' : null,
    },
  });

  const editModelFormSubmitHandler = useCallback(
    (values: CheckpointModelConfig) => {
      const responseBody = {
        base_model: model.base_model,
        model_name: model.model_name,
        body: values,
      };
      updateMainModel(responseBody)
        .unwrap()
        .then((payload) => {
          checkpointEditForm.setValues(payload as CheckpointModelConfig);
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
          checkpointEditForm.reset();
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
    [
      checkpointEditForm,
      dispatch,
      model.base_model,
      model.model_name,
      t,
      updateMainModel,
    ]
  );

  return (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex justifyContent="space-between" alignItems="center">
        <Flex flexDirection="column">
          <Text fontSize="lg" fontWeight="bold">
            {model.model_name}
          </Text>
          <Text fontSize="sm" color="base.400">
            {MODEL_TYPE_MAP[model.base_model]} Model
          </Text>
        </Flex>
        {![''].includes(model.base_model) ? (
          <ModelConvert model={model} />
        ) : (
          <Badge
            sx={{
              p: 2,
              borderRadius: 4,
              bg: 'error.200',
              _dark: { bg: 'error.400' },
            }}
          >
            Conversion Not Supported
          </Badge>
        )}
      </Flex>
      <Divider />

      <Flex
        flexDirection="column"
        maxHeight={window.innerHeight - 270}
        overflowY="scroll"
      >
        <form
          onSubmit={checkpointEditForm.onSubmit((values) =>
            editModelFormSubmitHandler(values)
          )}
        >
          <Flex flexDirection="column" overflowY="scroll" gap={4}>
            <IAIMantineTextInput
              label={t('modelManager.name')}
              {...checkpointEditForm.getInputProps('model_name')}
            />
            <IAIMantineTextInput
              label={t('modelManager.description')}
              {...checkpointEditForm.getInputProps('description')}
            />
            <BaseModelSelect
              required
              {...checkpointEditForm.getInputProps('base_model')}
            />
            <ModelVariantSelect
              required
              {...checkpointEditForm.getInputProps('variant')}
            />
            <IAIMantineTextInput
              required
              label={t('modelManager.modelLocation')}
              {...checkpointEditForm.getInputProps('path')}
            />
            <IAIMantineTextInput
              label={t('modelManager.vaeLocation')}
              {...checkpointEditForm.getInputProps('vae')}
            />

            <Flex flexDirection="column" gap={2}>
              {!useCustomConfig ? (
                <CheckpointConfigsSelect
                  required
                  {...checkpointEditForm.getInputProps('config')}
                />
              ) : (
                <IAIMantineTextInput
                  required
                  label={t('modelManager.config')}
                  {...checkpointEditForm.getInputProps('config')}
                />
              )}
              <IAISimpleCheckbox
                isChecked={useCustomConfig}
                onChange={() => setUseCustomConfig(!useCustomConfig)}
                label="Use Custom Config"
              />
            </Flex>

            <IAIButton
              type="submit"
              isDisabled={isBusy || isLoading}
              isLoading={isLoading}
            >
              {t('modelManager.updateModel')}
            </IAIButton>
          </Flex>
        </form>
      </Flex>
    </Flex>
  );
}
