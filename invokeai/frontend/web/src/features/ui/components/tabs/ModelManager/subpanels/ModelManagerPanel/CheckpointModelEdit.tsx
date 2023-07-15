import { Divider, Flex, Text } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { makeToast } from 'app/components/Toaster';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { addToast } from 'features/system/store/systemSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  CheckpointModelConfigEntity,
  useUpdateMainModelsMutation,
} from 'services/api/endpoints/models';
import { CheckpointModelConfig } from 'services/api/types';
import ModelConvert from './ModelConvert';

const baseModelSelectData = [
  { value: 'sd-1', label: MODEL_TYPE_MAP['sd-1'] },
  { value: 'sd-2', label: MODEL_TYPE_MAP['sd-2'] },
];

const variantSelectData = [
  { value: 'normal', label: 'Normal' },
  { value: 'inpaint', label: 'Inpaint' },
  { value: 'depth', label: 'Depth' },
];

type CheckpointModelEditProps = {
  model: CheckpointModelConfigEntity;
};

export default function CheckpointModelEdit(props: CheckpointModelEditProps) {
  const isBusy = useAppSelector(selectIsBusy);

  const { model } = props;

  const [updateMainModel, { isLoading }] = useUpdateMainModelsMutation();

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
        .catch((error) => {
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
        <ModelConvert model={model} />
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
              label={t('modelManager.description')}
              {...checkpointEditForm.getInputProps('description')}
            />
            <IAIMantineSelect
              label={t('modelManager.baseModel')}
              data={baseModelSelectData}
              {...checkpointEditForm.getInputProps('base_model')}
            />
            <IAIMantineSelect
              label={t('modelManager.variant')}
              data={variantSelectData}
              {...checkpointEditForm.getInputProps('variant')}
            />
            <IAIMantineTextInput
              label={t('modelManager.modelLocation')}
              {...checkpointEditForm.getInputProps('path')}
            />
            <IAIMantineTextInput
              label={t('modelManager.vaeLocation')}
              {...checkpointEditForm.getInputProps('vae')}
            />
            <IAIMantineTextInput
              label={t('modelManager.config')}
              {...checkpointEditForm.getInputProps('config')}
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
    </Flex>
  );
}
