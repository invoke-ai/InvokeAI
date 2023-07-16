import { Flex } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { makeToast } from 'app/components/Toaster';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import { addToast } from 'features/system/store/systemSlice';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAddMainModelsMutation } from 'services/api/endpoints/models';
import { CheckpointModelConfig } from 'services/api/types';
import BaseModelSelect from '../shared/BaseModelSelect';
import CheckpointConfigsSelect from '../shared/CheckpointConfigsSelect';
import ModelVariantSelect from '../shared/ModelVariantSelect';

export default function ManualAddCheckpoint() {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const manualAddCheckpointForm = useForm<CheckpointModelConfig>({
    initialValues: {
      model_name: '',
      base_model: 'sd-1',
      model_type: 'main',
      path: '',
      description: '',
      model_format: 'checkpoint',
      error: undefined,
      vae: '',
      variant: 'normal',
      config: 'configs\\stable-diffusion\\v1-inference.yaml',
    },
  });

  const [addMainModel] = useAddMainModelsMutation();

  const [useCustomConfig, setUseCustomConfig] = useState<boolean>(false);

  const manualAddCheckpointFormHandler = (values: CheckpointModelConfig) => {
    addMainModel({
      body: values,
    })
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: `Model Added: ${values.model_name}`,
              status: 'success',
            })
          )
        );
        manualAddCheckpointForm.reset();
      })
      .catch((error) => {
        if (error) {
          dispatch(
            addToast(
              makeToast({
                title: 'Model Add Failed',
                status: 'error',
              })
            )
          );
        }
      });
  };

  return (
    <form
      onSubmit={manualAddCheckpointForm.onSubmit((v) =>
        manualAddCheckpointFormHandler(v)
      )}
      style={{ width: '100%' }}
    >
      <Flex flexDirection="column" gap={2}>
        <IAIMantineTextInput
          label="Model Name"
          required
          {...manualAddCheckpointForm.getInputProps('model_name')}
        />
        <BaseModelSelect
          {...manualAddCheckpointForm.getInputProps('base_model')}
        />
        <IAIMantineTextInput
          label="Model Location"
          required
          {...manualAddCheckpointForm.getInputProps('path')}
        />
        <IAIMantineTextInput
          label="Description"
          {...manualAddCheckpointForm.getInputProps('description')}
        />
        <IAIMantineTextInput
          label="VAE Location"
          {...manualAddCheckpointForm.getInputProps('vae')}
        />
        <ModelVariantSelect
          {...manualAddCheckpointForm.getInputProps('variant')}
        />
        <Flex flexDirection="column" width="100%" gap={2}>
          {!useCustomConfig ? (
            <CheckpointConfigsSelect
              width="100%"
              {...manualAddCheckpointForm.getInputProps('config')}
            />
          ) : (
            <IAIMantineTextInput
              required
              label="Custom Config File Location"
              {...manualAddCheckpointForm.getInputProps('config')}
            />
          )}
          <IAISimpleCheckbox
            isChecked={useCustomConfig}
            onChange={() => setUseCustomConfig(!useCustomConfig)}
            label="Use Custom Config"
          />
          <IAIButton mt={2} type="submit">
            {t('modelManager.addModel')}
          </IAIButton>
        </Flex>
      </Flex>
    </form>
  );
}
