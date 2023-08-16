import { Box } from '@chakra-ui/react';
import {
  useFieldData,
  useFieldTemplate,
} from 'features/nodes/hooks/useNodeData';
import { memo } from 'react';
import BooleanInputField from './fieldTypes/BooleanInputField';
import ClipInputField from './fieldTypes/ClipInputField';
import CollectionInputField from './fieldTypes/CollectionInputField';
import CollectionItemInputField from './fieldTypes/CollectionItemInputField';
import ColorInputField from './fieldTypes/ColorInputField';
import ConditioningInputField from './fieldTypes/ConditioningInputField';
import ControlInputField from './fieldTypes/ControlInputField';
import ControlNetModelInputField from './fieldTypes/ControlNetModelInputField';
import EnumInputField from './fieldTypes/EnumInputField';
import ImageCollectionInputField from './fieldTypes/ImageCollectionInputField';
import ImageInputField from './fieldTypes/ImageInputField';
import LatentsInputField from './fieldTypes/LatentsInputField';
import LoRAModelInputField from './fieldTypes/LoRAModelInputField';
import MainModelInputField from './fieldTypes/MainModelInputField';
import NumberInputField from './fieldTypes/NumberInputField';
import RefinerModelInputField from './fieldTypes/RefinerModelInputField';
import SDXLMainModelInputField from './fieldTypes/SDXLMainModelInputField';
import StringInputField from './fieldTypes/StringInputField';
import UnetInputField from './fieldTypes/UnetInputField';
import VaeInputField from './fieldTypes/VaeInputField';
import VaeModelInputField from './fieldTypes/VaeModelInputField';

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

  return <Box p={2}>Unknown field type: {field?.type}</Box>;
};

export default memo(InputFieldRenderer);
