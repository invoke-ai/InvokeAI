import {
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  FloatInputFieldTemplate,
  FloatInputFieldValue,
  IntegerInputFieldTemplate,
  IntegerInputFieldValue,
} from 'features/nodes/types/types';
import { FieldComponentProps } from './types';

export const NumberInputFieldComponent = (
  props: FieldComponentProps<
    IntegerInputFieldValue | FloatInputFieldValue,
    IntegerInputFieldTemplate | FloatInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const handleValueChanged = (_: string, value: number) => {
    dispatch(fieldValueChanged({ nodeId, fieldName: field.name, value }));
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
