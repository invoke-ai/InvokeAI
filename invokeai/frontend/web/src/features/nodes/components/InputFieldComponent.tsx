import { Box } from '@chakra-ui/react';
import { memo } from 'react';
import { InputFieldTemplate, InputFieldValue } from '../types/types';
import ArrayInputFieldComponent from './fields/ArrayInputFieldComponent';
import BooleanInputFieldComponent from './fields/BooleanInputFieldComponent';
import ClipInputFieldComponent from './fields/ClipInputFieldComponent';
import ColorInputFieldComponent from './fields/ColorInputFieldComponent';
import ConditioningInputFieldComponent from './fields/ConditioningInputFieldComponent';
import ControlInputFieldComponent from './fields/ControlInputFieldComponent';
import ControlNetModelInputFieldComponent from './fields/ControlNetModelInputFieldComponent';
import EnumInputFieldComponent from './fields/EnumInputFieldComponent';
import ImageCollectionInputFieldComponent from './fields/ImageCollectionInputFieldComponent';
import ImageInputFieldComponent from './fields/ImageInputFieldComponent';
import ItemInputFieldComponent from './fields/ItemInputFieldComponent';
import LatentsInputFieldComponent from './fields/LatentsInputFieldComponent';
import LoRAModelInputFieldComponent from './fields/LoRAModelInputFieldComponent';
import ModelInputFieldComponent from './fields/ModelInputFieldComponent';
import NumberInputFieldComponent from './fields/NumberInputFieldComponent';
import StringInputFieldComponent from './fields/StringInputFieldComponent';
import UnetInputFieldComponent from './fields/UnetInputFieldComponent';
import VaeInputFieldComponent from './fields/VaeInputFieldComponent';
import VaeModelInputFieldComponent from './fields/VaeModelInputFieldComponent';
import RefinerModelInputFieldComponent from './fields/RefinerModelInputFieldComponent';

type InputFieldComponentProps = {
  nodeId: string;
  field: InputFieldValue;
  template: InputFieldTemplate;
};

// build an individual input element based on the schema
const InputFieldComponent = (props: InputFieldComponentProps) => {
  const { nodeId, field, template } = props;
  const { type } = field;

  if (type === 'string' && template.type === 'string') {
    return (
      <StringInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'boolean' && template.type === 'boolean') {
    return (
      <BooleanInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (
    (type === 'integer' && template.type === 'integer') ||
    (type === 'float' && template.type === 'float')
  ) {
    return (
      <NumberInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'enum' && template.type === 'enum') {
    return (
      <EnumInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'image' && template.type === 'image') {
    return (
      <ImageInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'latents' && template.type === 'latents') {
    return (
      <LatentsInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'conditioning' && template.type === 'conditioning') {
    return (
      <ConditioningInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'unet' && template.type === 'unet') {
    return (
      <UnetInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'clip' && template.type === 'clip') {
    return (
      <ClipInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'vae' && template.type === 'vae') {
    return (
      <VaeInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'control' && template.type === 'control') {
    return (
      <ControlInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'model' && template.type === 'model') {
    return (
      <ModelInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'refiner_model' && template.type === 'refiner_model') {
    return (
      <RefinerModelInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'vae_model' && template.type === 'vae_model') {
    return (
      <VaeModelInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'lora_model' && template.type === 'lora_model') {
    return (
      <LoRAModelInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'controlnet_model' && template.type === 'controlnet_model') {
    return (
      <ControlNetModelInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'array' && template.type === 'array') {
    return (
      <ArrayInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'item' && template.type === 'item') {
    return (
      <ItemInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'color' && template.type === 'color') {
    return (
      <ColorInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'item' && template.type === 'item') {
    return (
      <ItemInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  if (type === 'image_collection' && template.type === 'image_collection') {
    return (
      <ImageCollectionInputFieldComponent
        nodeId={nodeId}
        field={field}
        template={template}
      />
    );
  }

  return <Box p={2}>Unknown field type: {type}</Box>;
};

export default memo(InputFieldComponent);
