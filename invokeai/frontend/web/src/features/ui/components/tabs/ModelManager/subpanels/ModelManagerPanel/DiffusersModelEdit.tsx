import { Divider, Flex, Text } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { makeToast } from 'features/system/util/makeToast';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  DiffusersModelConfigEntity,
  useUpdateMainModelsMutation,
} from 'services/api/endpoints/models';
import { DiffusersModelConfig } from 'services/api/types';
import BaseModelSelect from '../shared/BaseModelSelect';
import ModelVariantSelect from '../shared/ModelVariantSelect';

type DiffusersModelEditProps = {
  model: DiffusersModelConfigEntity;
};

export default function DiffusersModelEdit(props: DiffusersModelEditProps) {
  const isBusy = useAppSelector(selectIsBusy);

  const { model } = props;

  const [updateMainModel, { isLoading }] = useUpdateMainModelsMutation();

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const diffusersEditForm = useForm<DiffusersModelConfig>({
    initialValues: {
      model_name: model.model_name ? model.model_name : '',
      base_model: model.base_model,
      model_type: 'main',
      path: model.path ? model.path : '',
      description: model.description ? model.description : '',
      model_format: 'diffusers',
      vae: model.vae ? model.vae : '',
      variant: model.variant,
    },
    validate: {
      path: (value) =>
        value.trim().length === 0 ? 'Must provide a path' : null,
    },
  });

  const editModelFormSubmitHandler = useCallback(
    (values: DiffusersModelConfig) => {
      const responseBody = {
        base_model: model.base_model,
        model_name: model.model_name,
        body: values,
      };

      updateMainModel(responseBody)
        .unwrap()
        .then((payload) => {
          diffusersEditForm.setValues(payload as DiffusersModelConfig);
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
          diffusersEditForm.reset();
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
      diffusersEditForm,
      dispatch,
      model.base_model,
      model.model_name,
      t,
      updateMainModel,
    ]
  );

  return (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex flexDirection="column">
        <Text fontSize="lg" fontWeight="bold">
          {model.model_name}
        </Text>
        <Text fontSize="sm" color="base.400">
          {MODEL_TYPE_MAP[model.base_model]} Model
        </Text>
      </Flex>
      <Divider />

      <form
        onSubmit={diffusersEditForm.onSubmit((values) =>
          editModelFormSubmitHandler(values)
        )}
      >
        <Flex flexDirection="column" overflowY="scroll" gap={4}>
          <IAIMantineTextInput
            label={t('modelManager.name')}
            {...diffusersEditForm.getInputProps('model_name')}
          />
          <IAIMantineTextInput
            label={t('modelManager.description')}
            {...diffusersEditForm.getInputProps('description')}
          />
          <BaseModelSelect
            required
            {...diffusersEditForm.getInputProps('base_model')}
          />
          <ModelVariantSelect
            required
            {...diffusersEditForm.getInputProps('variant')}
          />
          <IAIMantineTextInput
            required
            label={t('modelManager.modelLocation')}
            {...diffusersEditForm.getInputProps('path')}
          />
          <IAIMantineTextInput
            label={t('modelManager.vaeLocation')}
            {...diffusersEditForm.getInputProps('vae')}
          />
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
  );
}
