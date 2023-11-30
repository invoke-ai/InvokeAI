import { Divider, Flex, Text } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import {
  LORA_MODEL_FORMAT_MAP,
  MODEL_TYPE_MAP,
} from 'features/parameters/types/constants';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  LoRAModelConfigEntity,
  useUpdateLoRAModelsMutation,
} from 'services/api/endpoints/models';
import { LoRAModelConfig } from 'services/api/types';
import BaseModelSelect from 'features/modelManager/subpanels/shared/BaseModelSelect';

type LoRAModelEditProps = {
  model: LoRAModelConfigEntity;
};

export default function LoRAModelEdit(props: LoRAModelEditProps) {
  const { model } = props;

  const [updateLoRAModel, { isLoading }] = useUpdateLoRAModelsMutation();

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const loraEditForm = useForm<LoRAModelConfig>({
    initialValues: {
      model_name: model.model_name ? model.model_name : '',
      base_model: model.base_model,
      model_type: 'lora',
      path: model.path ? model.path : '',
      description: model.description ? model.description : '',
      model_format: model.model_format,
    },
    validate: {
      path: (value) =>
        value.trim().length === 0 ? 'Must provide a path' : null,
    },
  });

  const editModelFormSubmitHandler = useCallback(
    (values: LoRAModelConfig) => {
      const responseBody = {
        base_model: model.base_model,
        model_name: model.model_name,
        body: values,
      };

      updateLoRAModel(responseBody)
        .unwrap()
        .then((payload) => {
          loraEditForm.setValues(payload as LoRAModelConfig);
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
          loraEditForm.reset();
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
      dispatch,
      loraEditForm,
      model.base_model,
      model.model_name,
      t,
      updateLoRAModel,
    ]
  );

  return (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex flexDirection="column">
        <Text fontSize="lg" fontWeight="bold">
          {model.model_name}
        </Text>
        <Text fontSize="sm" color="base.400">
          {MODEL_TYPE_MAP[model.base_model]} {t('modelManager.model')} ⋅{' '}
          {LORA_MODEL_FORMAT_MAP[model.model_format]} {t('common.format')}
        </Text>
      </Flex>
      <Divider />

      <form
        onSubmit={loraEditForm.onSubmit((values) =>
          editModelFormSubmitHandler(values)
        )}
      >
        <Flex flexDirection="column" overflowY="scroll" gap={4}>
          <IAIMantineTextInput
            label={t('modelManager.name')}
            {...loraEditForm.getInputProps('model_name')}
          />
          <IAIMantineTextInput
            label={t('modelManager.description')}
            {...loraEditForm.getInputProps('description')}
          />
          <BaseModelSelect {...loraEditForm.getInputProps('base_model')} />
          <IAIMantineTextInput
            label={t('modelManager.modelLocation')}
            {...loraEditForm.getInputProps('path')}
          />
          <IAIButton type="submit" isLoading={isLoading}>
            {t('modelManager.updateModel')}
          </IAIButton>
        </Flex>
      </form>
    </Flex>
  );
}
