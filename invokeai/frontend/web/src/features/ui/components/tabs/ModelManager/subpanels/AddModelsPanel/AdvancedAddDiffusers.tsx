import { Flex } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useTranslation } from 'react-i18next';
import { useAddMainModelsMutation } from 'services/api/endpoints/models';
import { DiffusersModelConfig } from 'services/api/types';
import { setAdvancedAddScanModel } from '../../store/modelManagerSlice';
import BaseModelSelect from '../shared/BaseModelSelect';
import ModelVariantSelect from '../shared/ModelVariantSelect';
import { getModelName } from './util';

type AdvancedAddDiffusersProps = {
  model_path?: string;
};

export default function AdvancedAddDiffusers(props: AdvancedAddDiffusersProps) {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { model_path } = props;

  const [addMainModel] = useAddMainModelsMutation();

  const advancedAddDiffusersForm = useForm<DiffusersModelConfig>({
    initialValues: {
      model_name: model_path ? getModelName(model_path, false) : '',
      base_model: 'sd-1',
      model_type: 'main',
      path: model_path ? model_path : '',
      description: '',
      model_format: 'diffusers',
      error: undefined,
      vae: '',
      variant: 'normal',
    },
  });

  const advancedAddDiffusersFormHandler = (values: DiffusersModelConfig) => {
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
        advancedAddDiffusersForm.reset();
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
      onSubmit={advancedAddDiffusersForm.onSubmit((v) =>
        advancedAddDiffusersFormHandler(v)
      )}
      style={{ width: '100%' }}
    >
      <Flex flexDirection="column" gap={2}>
        <IAIMantineTextInput
          required
          label="Model Name"
          {...advancedAddDiffusersForm.getInputProps('model_name')}
        />
        <BaseModelSelect
          {...advancedAddDiffusersForm.getInputProps('base_model')}
        />
        <IAIMantineTextInput
          required
          label="Model Location"
          placeholder="Provide the path to a local folder where your Diffusers Model is stored"
          {...advancedAddDiffusersForm.getInputProps('path')}
          onBlur={(e) => {
            if (advancedAddDiffusersForm.values['model_name'] === '') {
              const modelName = getModelName(e.currentTarget.value, false);
              if (modelName) {
                advancedAddDiffusersForm.setFieldValue(
                  'model_name',
                  modelName as string
                );
              }
            }
          }}
        />
        <IAIMantineTextInput
          label="Description"
          {...advancedAddDiffusersForm.getInputProps('description')}
        />
        <IAIMantineTextInput
          label="VAE Location"
          {...advancedAddDiffusersForm.getInputProps('vae')}
        />
        <ModelVariantSelect
          {...advancedAddDiffusersForm.getInputProps('variant')}
        />
        <IAIButton mt={2} type="submit">
          {t('modelManager.addModel')}
        </IAIButton>
      </Flex>
    </form>
  );
}
