import {
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import { IntegerInputField, FloatInputField } from 'features/nodes/types';
import { FieldComponentProps } from './types';

export const NumberInputFieldComponent = (
  props: FieldComponentProps<IntegerInputField | FloatInputField>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const handleValueChanged = (_: string, value: number) => {
    dispatch(fieldValueChanged({ nodeId, fieldId: field.name, value }));
  };

  return (
    <NumberInput onChange={handleValueChanged} value={field.value}>
      <NumberInputField />
      <NumberInputStepper>
        <NumberIncrementStepper />
        <NumberDecrementStepper />
      </NumberInputStepper>
    </NumberInput>
  );
};
