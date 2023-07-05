import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';

import { Divider, Flex, Text } from '@chakra-ui/react';

// import { addNewModel } from 'app/socketio/actions';
import { useTranslation } from 'react-i18next';

import { useForm } from '@mantine/form';
import type { RootState } from 'app/store/store';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { MODEL_TYPE_MAP } from 'features/system/components/ModelSelect';
import { S } from 'services/api/types';

type DiffusersModel =
  | S<'StableDiffusion1ModelDiffusersConfig'>
  | S<'StableDiffusion2ModelDiffusersConfig'>;

type DiffusersModelEditProps = {
  modelToEdit: string;
  retrievedModel: DiffusersModel;
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

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const diffusersEditForm = useForm({
    initialValues: {
      name: retrievedModel.name,
      base_model: retrievedModel.base_model,
      type: 'main',
      path: retrievedModel.path,
      description: retrievedModel.description,
      model_format: 'diffusers',
      vae: retrievedModel.vae,
      variant: retrievedModel.variant,
    },
  });

  const editModelFormSubmitHandler = (values) => {
    console.log(values);
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
            label={t('modelManager.name')}
            {...diffusersEditForm.getInputProps('name')}
          />
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
