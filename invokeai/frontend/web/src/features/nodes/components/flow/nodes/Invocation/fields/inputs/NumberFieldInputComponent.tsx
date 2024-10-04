import { CompositeNumberInput } from '@invoke-ai/ui-library';
import { NUMPY_RAND_MAX } from 'app/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldNumberValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  FloatFieldInputInstance,
  FloatFieldInputTemplate,
  IntegerFieldInputInstance,
  IntegerFieldInputTemplate,
} from 'features/nodes/types/field';
import { isNil } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';

import type { FieldComponentProps } from './types';

const NumberFieldInputComponent = (
  props: FieldComponentProps<
    IntegerFieldInputInstance | FloatFieldInputInstance,
    IntegerFieldInputTemplate | FloatFieldInputTemplate
  >
) => {
  const { nodeId, field, fieldTemplate } = props;
  const dispatch = useAppDispatch();
  const isIntegerField = useMemo(() => fieldTemplate.type.name === 'IntegerField', [fieldTemplate.type]);

  const handleValueChanged = useCallback(
    (v: number) => {
      dispatch(
        fieldNumberValueChanged({
          nodeId,
          fieldName: field.name,
          value: isIntegerField ? Math.floor(Number(v)) : Number(v),
        })
      );
    },
    [dispatch, field.name, isIntegerField, nodeId]
  );

  const min = useMemo(() => {
    let min = -NUMPY_RAND_MAX;
    if (!isNil(fieldTemplate.minimum)) {
      min = fieldTemplate.minimum;
    }
    if (!isNil(fieldTemplate.exclusiveMinimum)) {
      min = fieldTemplate.exclusiveMinimum + 0.01;
    }
    return min;
  }, [fieldTemplate.exclusiveMinimum, fieldTemplate.minimum]);

  const max = useMemo(() => {
    let max = NUMPY_RAND_MAX;
    if (!isNil(fieldTemplate.maximum)) {
      max = fieldTemplate.maximum;
    }
    if (!isNil(fieldTemplate.exclusiveMaximum)) {
      max = fieldTemplate.exclusiveMaximum - 0.01;
    }
    return max;
  }, [fieldTemplate.exclusiveMaximum, fieldTemplate.maximum]);

  const step = useMemo(() => {
    if (isNil(fieldTemplate.multipleOf)) {
      return isIntegerField ? 1 : 0.1;
    }
    return fieldTemplate.multipleOf;
  }, [fieldTemplate.multipleOf, isIntegerField]);

  const fineStep = useMemo(() => {
    if (isNil(fieldTemplate.multipleOf)) {
      return isIntegerField ? 1 : 0.01;
    }
    return fieldTemplate.multipleOf;
  }, [fieldTemplate.multipleOf, isIntegerField]);

  return (
    <CompositeNumberInput
      defaultValue={fieldTemplate.default}
      onChange={handleValueChanged}
      value={field.value}
      min={min}
      max={max}
      step={step}
      fineStep={fineStep}
      className="nodrag"
    />
  );
};

export default memo(NumberFieldInputComponent);
