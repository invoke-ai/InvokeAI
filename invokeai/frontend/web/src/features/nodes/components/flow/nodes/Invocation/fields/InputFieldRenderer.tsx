import { Box, Text } from '@chakra-ui/react';
import { useFieldData } from 'features/nodes/hooks/useFieldData';
import { useFieldTemplate } from 'features/nodes/hooks/useFieldTemplate';
import { memo } from 'react';
import BooleanInputField from './inputs/BooleanInputField';
import ColorInputField from './inputs/ColorInputField';
import ControlNetModelInputField from './inputs/ControlNetModelInputField';
import EnumInputField from './inputs/EnumInputField';
import ImageInputField from './inputs/ImageInputField';
import LoRAModelInputField from './inputs/LoRAModelInputField';
import MainModelInputField from './inputs/MainModelInputField';
import NumberInputField from './inputs/NumberInputField';
import RefinerModelInputField from './inputs/RefinerModelInputField';
import SDXLMainModelInputField from './inputs/SDXLMainModelInputField';
import SchedulerInputField from './inputs/SchedulerInputField';
import StringInputField from './inputs/StringInputField';
import VaeModelInputField from './inputs/VaeModelInputField';
import IPAdapterModelInputField from './inputs/IPAdapterModelInputField';
import T2IAdapterModelInputField from './inputs/T2IAdapterModelInputField';
import BoardInputField from './inputs/BoardInputField';
import { useTranslation } from 'react-i18next';

type InputFieldProps = {
  nodeId: string;
  fieldName: string;
};

const InputFieldRenderer = ({ nodeId, fieldName }: InputFieldProps) => {
  const { t } = useTranslation();
  const field = useFieldData(nodeId, fieldName);
  const fieldTemplate = useFieldTemplate(nodeId, fieldName, 'input');

  if (fieldTemplate?.fieldKind === 'output') {
    return (
      <Box p={2}>
        {t('nodes.outputFieldInInput')}: {field?.type}
      </Box>
    );
  }

  if (
    (field?.type === 'string' && fieldTemplate?.type === 'string') ||
    (field?.type === 'StringPolymorphic' &&
      fieldTemplate?.type === 'StringPolymorphic')
  ) {
    return (
      <StringInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    (field?.type === 'boolean' && fieldTemplate?.type === 'boolean') ||
    (field?.type === 'BooleanPolymorphic' &&
      fieldTemplate?.type === 'BooleanPolymorphic')
  ) {
    return (
      <BooleanInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    (field?.type === 'integer' && fieldTemplate?.type === 'integer') ||
    (field?.type === 'float' && fieldTemplate?.type === 'float') ||
    (field?.type === 'FloatPolymorphic' &&
      fieldTemplate?.type === 'FloatPolymorphic') ||
    (field?.type === 'IntegerPolymorphic' &&
      fieldTemplate?.type === 'IntegerPolymorphic')
  ) {
    return (
      <NumberInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (field?.type === 'enum' && fieldTemplate?.type === 'enum') {
    return (
      <EnumInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    (field?.type === 'ImageField' && fieldTemplate?.type === 'ImageField') ||
    (field?.type === 'ImagePolymorphic' &&
      fieldTemplate?.type === 'ImagePolymorphic')
  ) {
    return (
      <ImageInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (field?.type === 'BoardField' && fieldTemplate?.type === 'BoardField') {
    return (
      <BoardInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'MainModelField' &&
    fieldTemplate?.type === 'MainModelField'
  ) {
    return (
      <MainModelInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'SDXLRefinerModelField' &&
    fieldTemplate?.type === 'SDXLRefinerModelField'
  ) {
    return (
      <RefinerModelInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'VaeModelField' &&
    fieldTemplate?.type === 'VaeModelField'
  ) {
    return (
      <VaeModelInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'LoRAModelField' &&
    fieldTemplate?.type === 'LoRAModelField'
  ) {
    return (
      <LoRAModelInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'ControlNetModelField' &&
    fieldTemplate?.type === 'ControlNetModelField'
  ) {
    return (
      <ControlNetModelInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'IPAdapterModelField' &&
    fieldTemplate?.type === 'IPAdapterModelField'
  ) {
    return (
      <IPAdapterModelInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'T2IAdapterModelField' &&
    fieldTemplate?.type === 'T2IAdapterModelField'
  ) {
    return (
      <T2IAdapterModelInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }
  if (field?.type === 'ColorField' && fieldTemplate?.type === 'ColorField') {
    return (
      <ColorInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'SDXLMainModelField' &&
    fieldTemplate?.type === 'SDXLMainModelField'
  ) {
    return (
      <SDXLMainModelInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (field?.type === 'Scheduler' && fieldTemplate?.type === 'Scheduler') {
    return (
      <SchedulerInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (field && fieldTemplate) {
    // Fallback for when there is no component for the type
    return null;
  }

  return (
    <Box p={1}>
      <Text
        sx={{
          fontSize: 'sm',
          fontWeight: 600,
          color: 'error.400',
          _dark: { color: 'error.300' },
        }}
      >
        {t('nodes.unknownFieldType')}: {field?.type}
      </Text>
    </Box>
  );
};

export default memo(InputFieldRenderer);
