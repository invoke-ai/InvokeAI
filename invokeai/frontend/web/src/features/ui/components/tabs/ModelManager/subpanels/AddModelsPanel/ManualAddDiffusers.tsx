import { Flex } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { makeToast } from 'app/components/Toaster';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import { addToast } from 'features/system/store/systemSlice';
import { useTranslation } from 'react-i18next';
import { useAddMainModelsMutation } from 'services/api/endpoints/models';
import { DiffusersModelConfig } from 'services/api/types';
import BaseModelSelect from '../shared/BaseModelSelect';
import ModelVariantSelect from '../shared/ModelVariantSelect';

export default function ManualAddDiffusers() {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const [addMainModel] = useAddMainModelsMutation();

  const manualAddDiffusersForm = useForm<DiffusersModelConfig>({
    initialValues: {
      model_name: '',
      base_model: 'sd-1',
      model_type: 'main',
      path: '',
      description: '',
      model_format: 'diffusers',
      error: undefined,
      vae: '',
      variant: 'normal',
    },
  });
  const manualAddDiffusersFormHandler = (values: DiffusersModelConfig) => {
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
        manualAddDiffusersForm.reset();
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
      onSubmit={manualAddDiffusersForm.onSubmit((v) =>
        manualAddDiffusersFormHandler(v)
      )}
      style={{ width: '100%' }}
    >
      <Flex flexDirection="column" gap={2}>
        <IAIMantineTextInput
          required
          label="Model Name"
          {...manualAddDiffusersForm.getInputProps('model_name')}
        />
        <BaseModelSelect
          {...manualAddDiffusersForm.getInputProps('base_model')}
        />
        <IAIMantineTextInput
          required
          label="Model Location"
          placeholder="Provide the path to a local folder where your Diffusers Model is stored"
          {...manualAddDiffusersForm.getInputProps('path')}
        />
        <IAIMantineTextInput
          label="Description"
          {...manualAddDiffusersForm.getInputProps('description')}
        />
        <IAIMantineTextInput
          label="VAE Location"
          {...manualAddDiffusersForm.getInputProps('vae')}
        />
        <ModelVariantSelect
          {...manualAddDiffusersForm.getInputProps('variant')}
        />
        <IAIButton mt={2} type="submit">
          {t('modelManager.addModel')}
        </IAIButton>
      </Flex>
    </form>
  );
}
