import { InputField } from '../types';
import { BooleanInputFieldComponent } from './fields/BooleanInputFieldComponent';
import { EnumInputFieldComponent } from './fields/EnumInputFieldComponent';
import { ImageInputFieldComponent } from './fields/ImageInputFieldComponent';
import { LatentsInputFieldComponent } from './fields/LatentsInputFieldComponent';
import { NumberInputFieldComponent } from './fields/NumberInputFieldComponent';
import { StringInputFieldComponent } from './fields/StringInputFieldComponent';

type InputFieldComponentProps = {
  nodeId: string;
  field: InputField;
};

// build an individual input element based on the schema
export const InputFieldComponent = (props: InputFieldComponentProps) => {
  const { nodeId, field } = props;
  const { type, value } = field;

  if (type === 'string') {
    return <StringInputFieldComponent nodeId={nodeId} field={field} />;
  }

  if (type === 'boolean') {
    return <BooleanInputFieldComponent nodeId={nodeId} field={field} />;
  }

  if (type === 'integer' || type === 'float') {
    return <NumberInputFieldComponent nodeId={nodeId} field={field} />;
  }

  if (type === 'enum') {
    return <EnumInputFieldComponent nodeId={nodeId} field={field} />;
  }

  if (type === 'image') {
    return <ImageInputFieldComponent nodeId={nodeId} field={field} />;
  }

  if (type === 'latents') {
    return <LatentsInputFieldComponent nodeId={nodeId} field={field} />;
  }
};
