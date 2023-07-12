import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';

import { Divider, Flex, Text } from '@chakra-ui/react';

// import { addNewModel } from 'app/socketio/actions';
import { useTranslation } from 'react-i18next';

import { useForm } from '@mantine/form';
import { makeToast } from 'app/components/Toaster';
import type { RootState } from 'app/store/store';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
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

  const [updateMainModel, { error }] = useUpdateMainModelsMutation();

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const diffusersEditForm = useForm<DiffusersModelConfig>({
    initialValues: {
      name: retrievedModel.name ? retrievedModel.name : '',
      base_model: retrievedModel.base_model,
      type: 'main',
      path: retrievedModel.path ? retrievedModel.path : '',
      description: retrievedModel.description ? retrievedModel.description : '',
      model_format: 'diffusers',
      vae: retrievedModel.vae ? retrievedModel.vae : '',
      variant: retrievedModel.variant,
    },
  });

  const editModelFormSubmitHandler = (values: DiffusersModelConfig) => {
    const responseBody = {
      base_model: retrievedModel.base_model,
      model_name: retrievedModel.name,
      body: values,
    };
    updateMainModel(responseBody);

    if (error) {
      dispatch(
        addToast(
          makeToast({
            title: t('modelManager.modelUpdateFailed'),
            status: 'success',
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
          {retrievedModel.name}
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
          <IAIInput
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
          <IAIInput
            label={t('modelManager.modelLocation')}
            {...diffusersEditForm.getInputProps('path')}
          />
          <IAIInput
            label={t('modelManager.vaeLocation')}
            {...diffusersEditForm.getInputProps('vae')}
          />
          <IAIButton disabled={isProcessing} type="submit">
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
