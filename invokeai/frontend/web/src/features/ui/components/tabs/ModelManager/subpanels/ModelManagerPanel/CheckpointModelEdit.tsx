import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';

import { Divider, Flex, Text } from '@chakra-ui/react';

// import { addNewModel } from 'app/socketio/actions';
import { useForm } from '@mantine/form';
import { useTranslation } from 'react-i18next';

import type { RootState } from 'app/store/store';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { MODEL_TYPE_MAP } from 'features/system/components/ModelSelect';
import { S } from 'services/api/types';
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

export type CheckpointModel =
  | S<'StableDiffusion1ModelCheckpointConfig'>
  | S<'StableDiffusion2ModelCheckpointConfig'>;

type CheckpointModelEditProps = {
  modelToEdit: string;
  retrievedModel: CheckpointModel;
};

export default function CheckpointModelEdit(props: CheckpointModelEditProps) {
  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const { modelToEdit, retrievedModel } = props;

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const checkpointEditForm = useForm({
    initialValues: {
      name: retrievedModel.name,
      base_model: retrievedModel.base_model,
      type: 'main',
      path: retrievedModel.path,
      description: retrievedModel.description,
      model_format: 'checkpoint',
      vae: retrievedModel.vae,
      config: retrievedModel.config,
      variant: retrievedModel.variant,
    },
  });

  const editModelFormSubmitHandler = (values) => {
    console.log(values);
  };

  return modelToEdit ? (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex justifyContent="space-between" alignItems="center">
        <Flex flexDirection="column">
          <Text fontSize="lg" fontWeight="bold">
            {retrievedModel.name}
          </Text>
          <Text fontSize="sm" color="base.400">
            {MODEL_TYPE_MAP[retrievedModel.base_model]} Model
          </Text>
        </Flex>
        <ModelConvert model={retrievedModel} />
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
            <IAIInput
              label={t('modelManager.name')}
              {...checkpointEditForm.getInputProps('name')}
            />
            <IAIInput
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
            <IAIInput
              label={t('modelManager.modelLocation')}
              {...checkpointEditForm.getInputProps('path')}
            />
            <IAIInput
              label={t('modelManager.vaeLocation')}
              {...checkpointEditForm.getInputProps('vae')}
            />
            <IAIInput
              label={t('modelManager.config')}
              {...checkpointEditForm.getInputProps('config')}
            />
            <IAIButton disabled={isProcessing} type="submit">
              {t('modelManager.updateModel')}
            </IAIButton>
          </Flex>
        </form>
      </Flex>
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
      <Text fontWeight={500}>Pick A Model To Edit</Text>
    </Flex>
  );
}
