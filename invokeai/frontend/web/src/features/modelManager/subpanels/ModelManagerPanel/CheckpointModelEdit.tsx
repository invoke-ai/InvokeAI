import { Badge, Divider, Flex } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
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
import { useCallback, useEffect, useState } from 'react';
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

export default function CheckpointModelEdit(props: CheckpointModelEditProps) {
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

  const handleChangeUseCustomConfig = useCallback(
    () => setUseCustomConfig((prev) => !prev),
    []
  );

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
        <form
          onSubmit={checkpointEditForm.onSubmit((values) =>
            editModelFormSubmitHandler(values)
          )}
        >
          <Flex flexDirection="column" overflowY="scroll" gap={4}>
            <InvControl label={t('modelManager.name')}>
              <InvInput {...checkpointEditForm.getInputProps('model_name')} />
            </InvControl>
            <InvControl label={t('modelManager.description')}>
              <InvInput {...checkpointEditForm.getInputProps('description')} />
            </InvControl>
            <BaseModelSelect
              required
              {...checkpointEditForm.getInputProps('base_model')}
            />
            <ModelVariantSelect
              required
              {...checkpointEditForm.getInputProps('variant')}
            />
            <InvControl isRequired label={t('modelManager.modelLocation')}>
              <InvInput {...checkpointEditForm.getInputProps('path')} />
            </InvControl>
            <InvControl label={t('modelManager.vaeLocation')}>
              <InvInput {...checkpointEditForm.getInputProps('vae')} />
            </InvControl>

            <Flex flexDirection="column" gap={2}>
              {!useCustomConfig ? (
                <CheckpointConfigsSelect
                  required
                  {...checkpointEditForm.getInputProps('config')}
                />
              ) : (
                <InvControl isRequired label={t('modelManager.config')}>
                  <InvInput {...checkpointEditForm.getInputProps('config')} />
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
}
