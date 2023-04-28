import {
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  FloatInputFieldTemplate,
  FloatInputFieldValue,
  IntegerInputFieldTemplate,
  IntegerInputFieldValue,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { FieldComponentProps } from './types';

const NumberInputFieldComponent = (
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
    <NumberInput
      onChange={handleValueChanged}
      value={field.value}
      step={props.template.type === 'integer' ? 1 : 0.1}
      precision={props.template.type === 'integer' ? 0 : 3}
    >
      <NumberInputField />
      <NumberInputStepper>
        <NumberIncrementStepper />
        <NumberDecrementStepper />
      </NumberInputStepper>
    </NumberInput>
  );
};

export default memo(NumberInputFieldComponent);
