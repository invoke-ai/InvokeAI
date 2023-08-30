import { Box, Text } from '@chakra-ui/react';
import { useFieldData } from 'features/nodes/hooks/useFieldData';
import { useFieldTemplate } from 'features/nodes/hooks/useFieldTemplate';
import { memo } from 'react';
import BooleanInputField from './inputs/BooleanInputField';
import ClipInputField from './inputs/ClipInputField';
import CollectionInputField from './inputs/CollectionInputField';
import CollectionItemInputField from './inputs/CollectionItemInputField';
import ColorInputField from './inputs/ColorInputField';
import ConditioningInputField from './inputs/ConditioningInputField';
import ControlInputField from './inputs/ControlInputField';
import ControlNetModelInputField from './inputs/ControlNetModelInputField';
import DenoiseMaskInputField from './inputs/DenoiseMaskInputField';
import EnumInputField from './inputs/EnumInputField';
import ImageCollectionInputField from './inputs/ImageCollectionInputField';
import ImageInputField from './inputs/ImageInputField';
import LatentsInputField from './inputs/LatentsInputField';
import LoRAModelInputField from './inputs/LoRAModelInputField';
import MainModelInputField from './inputs/MainModelInputField';
import NumberInputField from './inputs/NumberInputField';
import RefinerModelInputField from './inputs/RefinerModelInputField';
import SDXLMainModelInputField from './inputs/SDXLMainModelInputField';
import SchedulerInputField from './inputs/SchedulerInputField';
import StringInputField from './inputs/StringInputField';
import UnetInputField from './inputs/UnetInputField';
import VaeInputField from './inputs/VaeInputField';
import VaeModelInputField from './inputs/VaeModelInputField';

type InputFieldProps = {
  nodeId: string;
  fieldName: string;
};

// build an individual input element based on the schema
const InputFieldRenderer = ({ nodeId, fieldName }: InputFieldProps) => {
  const field = useFieldData(nodeId, fieldName);
  const fieldTemplate = useFieldTemplate(nodeId, fieldName, 'input');

  if (fieldTemplate?.fieldKind === 'output') {
    return <Box p={2}>Output field in input: {field?.type}</Box>;
  }

  if (field?.type === 'string' && fieldTemplate?.type === 'string') {
    return (
      <StringInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (field?.type === 'boolean' && fieldTemplate?.type === 'boolean') {
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
    (field?.type === 'float' && fieldTemplate?.type === 'float')
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

  if (field?.type === 'ImageField' && fieldTemplate?.type === 'ImageField') {
    return (
      <ImageInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'LatentsField' &&
    fieldTemplate?.type === 'LatentsField'
  ) {
    return (
      <LatentsInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'DenoiseMaskField' &&
    fieldTemplate?.type === 'DenoiseMaskField'
  ) {
    return (
      <DenoiseMaskInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'ConditioningField' &&
    fieldTemplate?.type === 'ConditioningField'
  ) {
    return (
      <ConditioningInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (field?.type === 'UNetField' && fieldTemplate?.type === 'UNetField') {
    return (
      <UnetInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (field?.type === 'ClipField' && fieldTemplate?.type === 'ClipField') {
    return (
      <ClipInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (field?.type === 'VaeField' && fieldTemplate?.type === 'VaeField') {
    return (
      <VaeInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'ControlField' &&
    fieldTemplate?.type === 'ControlField'
  ) {
    return (
      <ControlInputField
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

  if (field?.type === 'Collection' && fieldTemplate?.type === 'Collection') {
    return (
      <CollectionInputField
        nodeId={nodeId}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    field?.type === 'CollectionItem' &&
    fieldTemplate?.type === 'CollectionItem'
  ) {
    return (
      <CollectionItemInputField
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
    field?.type === 'ImageCollection' &&
    fieldTemplate?.type === 'ImageCollection'
  ) {
    return (
      <ImageCollectionInputField
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
        Unknown field type: {field?.type}
      </Text>
    </Box>
  );
};

export default memo(InputFieldRenderer);
