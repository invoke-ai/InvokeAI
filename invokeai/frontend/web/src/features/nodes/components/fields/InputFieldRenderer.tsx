import { Box } from '@chakra-ui/react';
import { memo } from 'react';
import {
  InputFieldTemplate,
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
} from '../../types/types';
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
  nodeData: InvocationNodeData;
  nodeTemplate: InvocationTemplate;
  field: InputFieldValue;
  fieldTemplate: InputFieldTemplate;
};

// build an individual input element based on the schema
const InputFieldRenderer = (props: InputFieldProps) => {
  const { nodeData, nodeTemplate, field, fieldTemplate } = props;
  const { type } = field;

  if (type === 'string' && fieldTemplate.type === 'string') {
    return (
      <StringInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'boolean' && fieldTemplate.type === 'boolean') {
    return (
      <BooleanInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    (type === 'integer' && fieldTemplate.type === 'integer') ||
    (type === 'float' && fieldTemplate.type === 'float')
  ) {
    return (
      <NumberInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'enum' && fieldTemplate.type === 'enum') {
    return (
      <EnumInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'ImageField' && fieldTemplate.type === 'ImageField') {
    return (
      <ImageInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'LatentsField' && fieldTemplate.type === 'LatentsField') {
    return (
      <LatentsInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    type === 'ConditioningField' &&
    fieldTemplate.type === 'ConditioningField'
  ) {
    return (
      <ConditioningInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'UNetField' && fieldTemplate.type === 'UNetField') {
    return (
      <UnetInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'ClipField' && fieldTemplate.type === 'ClipField') {
    return (
      <ClipInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'VaeField' && fieldTemplate.type === 'VaeField') {
    return (
      <VaeInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'ControlField' && fieldTemplate.type === 'ControlField') {
    return (
      <ControlInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'MainModelField' && fieldTemplate.type === 'MainModelField') {
    return (
      <MainModelInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    type === 'SDXLRefinerModelField' &&
    fieldTemplate.type === 'SDXLRefinerModelField'
  ) {
    return (
      <RefinerModelInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'VaeModelField' && fieldTemplate.type === 'VaeModelField') {
    return (
      <VaeModelInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'LoRAModelField' && fieldTemplate.type === 'LoRAModelField') {
    return (
      <LoRAModelInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    type === 'ControlNetModelField' &&
    fieldTemplate.type === 'ControlNetModelField'
  ) {
    return (
      <ControlNetModelInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'Collection' && fieldTemplate.type === 'Collection') {
    return (
      <CollectionInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'CollectionItem' && fieldTemplate.type === 'CollectionItem') {
    return (
      <CollectionItemInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'ColorField' && fieldTemplate.type === 'ColorField') {
    return (
      <ColorInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (type === 'ImageCollection' && fieldTemplate.type === 'ImageCollection') {
    return (
      <ImageCollectionInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  if (
    type === 'SDXLMainModelField' &&
    fieldTemplate.type === 'SDXLMainModelField'
  ) {
    return (
      <SDXLMainModelInputField
        nodeData={nodeData}
        nodeTemplate={nodeTemplate}
        field={field}
        fieldTemplate={fieldTemplate}
      />
    );
  }

  return <Box p={2}>Unknown field type: {type}</Box>;
};

export default memo(InputFieldRenderer);
