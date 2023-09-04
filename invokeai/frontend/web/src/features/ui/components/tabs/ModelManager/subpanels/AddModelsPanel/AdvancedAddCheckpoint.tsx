import { Flex } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAddMainModelsMutation } from 'services/api/endpoints/models';
import { CheckpointModelConfig } from 'services/api/types';
import { setAdvancedAddScanModel } from '../../store/modelManagerSlice';
import BaseModelSelect from '../shared/BaseModelSelect';
import CheckpointConfigsSelect from '../shared/CheckpointConfigsSelect';
import ModelVariantSelect from '../shared/ModelVariantSelect';
import { getModelName } from './util';

type AdvancedAddCheckpointProps = {
  model_path?: string;
};

export default function AdvancedAddCheckpoint(
  props: AdvancedAddCheckpointProps
) {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { model_path } = props;

  const advancedAddCheckpointForm = useForm<CheckpointModelConfig>({
    initialValues: {
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
  });

  const [addMainModel] = useAddMainModelsMutation();

  const [useCustomConfig, setUseCustomConfig] = useState<boolean>(false);

  const advancedAddCheckpointFormHandler = (values: CheckpointModelConfig) => {
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
        advancedAddCheckpointForm.reset();

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
      onSubmit={advancedAddCheckpointForm.onSubmit((v) =>
        advancedAddCheckpointFormHandler(v)
      )}
      style={{ width: '100%' }}
    >
      <Flex flexDirection="column" gap={2}>
        <IAIMantineTextInput
          label="Model Name"
          required
          {...advancedAddCheckpointForm.getInputProps('model_name')}
        />
        <BaseModelSelect
          {...advancedAddCheckpointForm.getInputProps('base_model')}
        />
        <IAIMantineTextInput
          label="Model Location"
          required
          {...advancedAddCheckpointForm.getInputProps('path')}
          onBlur={(e) => {
            if (advancedAddCheckpointForm.values['model_name'] === '') {
              const modelName = getModelName(e.currentTarget.value);
              if (modelName) {
                advancedAddCheckpointForm.setFieldValue(
                  'model_name',
                  modelName as string
                );
              }
            }
          }}
        />
        <IAIMantineTextInput
          label="Description"
          {...advancedAddCheckpointForm.getInputProps('description')}
        />
        <IAIMantineTextInput
          label="VAE Location"
          {...advancedAddCheckpointForm.getInputProps('vae')}
        />
        <ModelVariantSelect
          {...advancedAddCheckpointForm.getInputProps('variant')}
        />
        <Flex flexDirection="column" width="100%" gap={2}>
          {!useCustomConfig ? (
            <CheckpointConfigsSelect
              required
              width="100%"
              {...advancedAddCheckpointForm.getInputProps('config')}
            />
          ) : (
            <IAIMantineTextInput
              required
              label="Custom Config File Location"
              {...advancedAddCheckpointForm.getInputProps('config')}
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
