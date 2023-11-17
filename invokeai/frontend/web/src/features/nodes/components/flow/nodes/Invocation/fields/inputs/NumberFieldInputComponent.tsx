import {
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { numberStringRegex } from 'common/components/IAINumberInput';
import { fieldNumberValueChanged } from 'features/nodes/store/nodesSlice';
import {
  FloatFieldInputInstance,
  FloatFieldInputTemplate,
  IntegerFieldInputInstance,
  IntegerFieldInputTemplate,
} from 'features/nodes/types/field';
import { FieldComponentProps } from './types';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';

const NumberFieldInputComponent = (
  props: FieldComponentProps<
    IntegerFieldInputInstance | FloatFieldInputInstance,
    IntegerFieldInputTemplate | FloatFieldInputTemplate
  >
) => {
  const { nodeId, field, fieldTemplate } = props;
  const dispatch = useAppDispatch();
  const [valueAsString, setValueAsString] = useState<string>(
    String(field.value)
  );
  const isIntegerField = useMemo(
    () => fieldTemplate.type.name === 'IntegerField',
    [fieldTemplate.type]
  );

  const handleValueChanged = useCallback(
    (v: string) => {
      setValueAsString(v);
      // This allows negatives and decimals e.g. '-123', `.5`, `-0.2`, etc.
      if (!v.match(numberStringRegex)) {
        // Cast the value to number. Floor it if it should be an integer.
        dispatch(
          fieldNumberValueChanged({
            nodeId,
            fieldName: field.name,
            value: isIntegerField ? Math.floor(Number(v)) : Number(v),
          })
        );
      }
    },
    [dispatch, field.name, isIntegerField, nodeId]
  );

  useEffect(() => {
    if (
      !valueAsString.match(numberStringRegex) &&
      field.value !== Number(valueAsString)
    ) {
      setValueAsString(String(field.value));
    }
  }, [field.value, valueAsString]);

  return (
    <NumberInput
      onChange={handleValueChanged}
      value={valueAsString}
      step={isIntegerField ? 1 : 0.1}
      precision={isIntegerField ? 0 : 3}
    >
      <NumberInputField className="nodrag" />
      <NumberInputStepper>
        <NumberIncrementStepper />
        <NumberDecrementStepper />
      </NumberInputStepper>
    </NumberInput>
  );
};

export default memo(NumberFieldInputComponent);
