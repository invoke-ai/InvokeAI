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
import { InputField } from '../types';

// build an individual input element based on the schema
export const buildFieldComponent = (
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
