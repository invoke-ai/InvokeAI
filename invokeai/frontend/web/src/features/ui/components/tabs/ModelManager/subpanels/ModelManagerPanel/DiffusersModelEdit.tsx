import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';

import { Divider, Flex, Text } from '@chakra-ui/react';

// import { addNewModel } from 'app/socketio/actions';
import { useTranslation } from 'react-i18next';

import { useForm } from '@mantine/form';
import { makeToast } from 'app/components/Toaster';
import type { RootState } from 'app/store/store';
import IAIButton from 'common/components/IAIButton';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { MODEL_TYPE_MAP } from 'features/system/components/ModelSelect';
import { addToast } from 'features/system/store/systemSlice';
import { useUpdateMainModelsMutation } from 'services/api/endpoints/models';
import { components } from 'services/api/schema';

export type DiffusersModelConfig =
  | components['schemas']['StableDiffusion1ModelDiffusersConfig']
  | components['schemas']['StableDiffusion2ModelDiffusersConfig'];

type DiffusersModelEditProps = {
  modelToEdit: string;
  retrievedModel: DiffusersModelConfig;
};

const baseModelSelectData = [
  { value: 'sd-1', label: MODEL_TYPE_MAP['sd-1'] },
  { value: 'sd-2', label: MODEL_TYPE_MAP['sd-2'] },
];

const variantSelectData = [
  { value: 'normal', label: 'Normal' },
  { value: 'inpaint', label: 'Inpaint' },
  { value: 'depth', label: 'Depth' },
];

export default function DiffusersModelEdit(props: DiffusersModelEditProps) {
  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );
  const { retrievedModel, modelToEdit } = props;

  const [updateMainModel, { isLoading, error }] = useUpdateMainModelsMutation();

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const diffusersEditForm = useForm<DiffusersModelConfig>({
    initialValues: {
      model_name: retrievedModel.model_name ? retrievedModel.model_name : '',
      base_model: retrievedModel.base_model,
      model_type: 'main',
      path: retrievedModel.path ? retrievedModel.path : '',
      description: retrievedModel.description ? retrievedModel.description : '',
      model_format: 'diffusers',
      vae: retrievedModel.vae ? retrievedModel.vae : '',
      variant: retrievedModel.variant,
    },
    validate: {
      path: (value) =>
        value.trim().length === 0 ? 'Must provide a path' : null,
    },
  });

  const editModelFormSubmitHandler = (values: DiffusersModelConfig) => {
    const responseBody = {
      base_model: retrievedModel.base_model,
      model_name: retrievedModel.model_name,
      body: values,
    };
    updateMainModel(responseBody);

    if (error) {
      dispatch(
        addToast(
          makeToast({
            title: t('modelManager.modelUpdateFailed'),
            status: 'error',
          })
        )
      );
    }

    dispatch(
      addToast(
        makeToast({
          title: t('modelManager.modelUpdated'),
          status: 'success',
        })
      )
    );
  };

  return modelToEdit ? (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex flexDirection="column">
        <Text fontSize="lg" fontWeight="bold">
          {retrievedModel.model_name}
        </Text>
        <Text fontSize="sm" color="base.400">
          {MODEL_TYPE_MAP[retrievedModel.base_model]} Model
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
            label={t('modelManager.description')}
            {...diffusersEditForm.getInputProps('description')}
          />
          <IAIMantineSelect
            label={t('modelManager.baseModel')}
            data={baseModelSelectData}
            {...diffusersEditForm.getInputProps('base_model')}
          />
          <IAIMantineSelect
            label={t('modelManager.variant')}
            data={variantSelectData}
            {...diffusersEditForm.getInputProps('variant')}
          />
          <IAIMantineTextInput
            label={t('modelManager.modelLocation')}
            {...diffusersEditForm.getInputProps('path')}
          />
          <IAIMantineTextInput
            label={t('modelManager.vaeLocation')}
            {...diffusersEditForm.getInputProps('vae')}
          />
          <IAIButton
            disabled={isProcessing}
            type="submit"
            isLoading={isLoading}
          >
            {t('modelManager.updateModel')}
          </IAIButton>
        </Flex>
      </form>
    </Flex>
  ) : (
    <Flex
      sx={{
        width: '100%',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: 'base',
        bg: 'base.900',
      }}
    >
      <Text fontWeight={'500'}>Pick A Model To Edit</Text>
    </Flex>
  );
}
