import {
  Input,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Select,
  Switch,
} from '@chakra-ui/react';
import { ReactNode } from 'react';
import { InputField, ProcessedNodeSchemaObject } from '../types';

// build an individual input element based on the schema
export const buildFieldComponent = (
  field: ProcessedNodeSchemaObject
): ReactNode => {
  if (field.fieldType === 'string') {
    // `string` fields may either be a text input or an enum ie select
    if (field.enum) {
      return (
        <Select defaultValue={field.default}>
          {field.enum?.map((option) => (
            <option key={option}>{option}</option>
          ))}
        </Select>
      );
    }

    return <Input defaultValue={field.default}></Input>;
  } else if (field.fieldType === 'boolean') {
    return <Switch defaultValue={field.default}></Switch>;
  } else if (['integer', 'number'].includes(field.fieldType as string)) {
    return (
      <NumberInput defaultValue={field.default}>
        <NumberInputField />
        <NumberInputStepper>
          <NumberIncrementStepper />
          <NumberDecrementStepper />
        </NumberInputStepper>
      </NumberInput>
    );
  }
};

// build an individual input element based on the schema
export const _buildFieldComponent = (
  nodeId: string,
  field: InputField
): ReactNode => {
  const { type } = field;

  if (type === 'string') {
    return <Input></Input>;
  }

  if (type === 'boolean') {
    return <Switch></Switch>;
  }

  if (['integer', 'number'].includes(type)) {
    return (
      <NumberInput>
        <NumberInputField />
        <NumberInputStepper>
          <NumberIncrementStepper />
          <NumberDecrementStepper />
        </NumberInputStepper>
      </NumberInput>
    );
  }

  // `string` fields may either be a text input or an enum ie select
  if (type === 'enum') {
    return (
      <Select>
        {field.options.map((option) => (
          <option key={option}>{option}</option>
        ))}
      </Select>
    );
  }
};
