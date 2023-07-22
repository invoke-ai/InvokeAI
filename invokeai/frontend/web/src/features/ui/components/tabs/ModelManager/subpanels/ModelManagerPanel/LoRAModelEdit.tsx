import { Divider, Flex, Text } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import {
  LORA_MODEL_FORMAT_MAP,
  MODEL_TYPE_MAP,
} from 'features/parameters/types/constants';
import { useTranslation } from 'react-i18next';
import { LoRAModelConfigEntity } from 'services/api/endpoints/models';
import { LoRAModelConfig } from 'services/api/types';
import BaseModelSelect from '../shared/BaseModelSelect';

type LoRAModelEditProps = {
  model: LoRAModelConfigEntity;
};

export default function LoRAModelEdit(props: LoRAModelEditProps) {
  const { model } = props;

  const { t } = useTranslation();

  const loraEditForm = useForm<LoRAModelConfig>({
    initialValues: {
      model_name: model.model_name ? model.model_name : '',
      base_model: model.base_model,
      model_type: 'lora',
      path: model.path ? model.path : '',
      description: model.description ? model.description : '',
      model_format: model.model_format,
    },
    validate: {
      path: (value) =>
        value.trim().length === 0 ? 'Must provide a path' : null,
    },
  });

  return (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex flexDirection="column">
        <Text fontSize="lg" fontWeight="bold">
          {model.model_name}
        </Text>
        <Text fontSize="sm" color="base.400">
          {MODEL_TYPE_MAP[model.base_model]} Model ⋅{' '}
          {LORA_MODEL_FORMAT_MAP[model.model_format]} format
        </Text>
      </Flex>
      <Divider />

      <form>
        <Flex flexDirection="column" overflowY="scroll" gap={4}>
          <IAIMantineTextInput
            label={t('modelManager.name')}
            readOnly={true}
            disabled={true}
            {...loraEditForm.getInputProps('model_name')}
          />
          <IAIMantineTextInput
            label={t('modelManager.description')}
            readOnly={true}
            disabled={true}
            {...loraEditForm.getInputProps('description')}
          />
          <BaseModelSelect
            readOnly={true}
            disabled={true}
            {...loraEditForm.getInputProps('base_model')}
          />
          <IAIMantineTextInput
            readOnly={true}
            disabled={true}
            label={t('modelManager.modelLocation')}
            {...loraEditForm.getInputProps('path')}
          />
          <Text color="base.400">
            {t('Editing LoRA model metadata is not yet supported.')}
          </Text>
        </Flex>
      </form>
    </Flex>
  );
}
